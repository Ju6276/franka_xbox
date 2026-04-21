import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Convert raw teleoperation episodes to LeRobot-style parquet dataset.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="dataset_raw",
        help="Directory containing raw episodes, e.g. dataset_raw/episode_000000",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="demo_data_converted",
        help="Output directory for converted dataset.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="franka_teleop",
        help="Task name written into meta files.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="franka",
        help="Robot type written into meta files.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="Dataset FPS.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_image_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def make_state_vector(step: Dict[str, Any]) -> List[float]:
    joint_pos = step.get("joint_pos", [])
    joint_vel = step.get("joint_vel", [])
    tcp_pos = step.get("tcp_pos", [])
    gripper_pos = step.get("gripper_pos", [])
    return list(joint_pos) + list(joint_vel) + list(tcp_pos) + list(gripper_pos)


def compute_episode_stats(df: pd.DataFrame) -> Dict[str, Any]:
    state_arr = np.stack(df["observation.state"].to_list(), axis=0)
    action_arr = np.stack(df["action"].to_list(), axis=0)

    stats = {
        "num_frames": int(len(df)),
        "state_mean": state_arr.mean(axis=0).tolist(),
        "state_std": state_arr.std(axis=0).tolist(),
        "state_min": state_arr.min(axis=0).tolist(),
        "state_max": state_arr.max(axis=0).tolist(),
        "action_mean": action_arr.mean(axis=0).tolist(),
        "action_std": action_arr.std(axis=0).tolist(),
        "action_min": action_arr.min(axis=0).tolist(),
        "action_max": action_arr.max(axis=0).tolist(),
    }
    return stats


def convert_episode(raw_episode_dir: Path, out_chunk_dir: Path, episode_index: int, fps: float) -> Dict[str, Any]:
    steps_path = raw_episode_dir / "steps.jsonl"
    if not steps_path.exists():
        raise FileNotFoundError(f"Missing {steps_path}")

    steps = load_jsonl(steps_path)
    if len(steps) == 0:
        raise ValueError(f"No steps found in {steps_path}")

    parquet_rows = []

    for i, step in enumerate(steps):
        row = {
            "episode_index": episode_index,
            "frame_index": int(step.get("frame_id", i)),
            "timestamp": float(step.get("timestamp", i / fps)),
            "action": step["action"],
            "observation.state": make_state_vector(step),
            "joint_command": step.get("joint_command"),
            "gripper_command": step.get("gripper_command"),
            "tcp_pos": step.get("tcp_pos"),
            "joint_pos": step.get("joint_pos"),
            "joint_vel": step.get("joint_vel"),
            "gripper_pos": step.get("gripper_pos"),
        }

        front_rel = step.get("front_path")
        wrist_rel = step.get("wrist_path")

        if front_rel is not None:
            front_path = raw_episode_dir / front_rel
            row["observation.image"] = {"bytes": read_image_bytes(front_path)}
        else:
            row["observation.image"] = None

        if wrist_rel is not None:
            wrist_path = raw_episode_dir / wrist_rel
            row["observation.wrist_image"] = {"bytes": read_image_bytes(wrist_path)}
        else:
            row["observation.wrist_image"] = None

        parquet_rows.append(row)

    df = pd.DataFrame(parquet_rows)
    parquet_path = out_chunk_dir / f"episode_{episode_index:06d}.parquet"
    df.to_parquet(parquet_path, index=False)

    ep_stats = compute_episode_stats(df)

    episode_meta = {
        "episode_index": episode_index,
        "length": int(len(df)),
        "parquet_path": str(parquet_path.relative_to(out_chunk_dir.parent.parent)),
        "fps": fps,
    }
    episode_stats_meta = {
        "episode_index": episode_index,
        **ep_stats,
    }
    return {
        "episode_meta": episode_meta,
        "episode_stats": episode_stats_meta,
        "num_frames": int(len(df)),
    }


def save_json(path: Path, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def aggregate_global_stats(episode_stats_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_frames = sum(r["num_frames"] for r in episode_stats_rows)
    return {
        "total_episodes": len(episode_stats_rows),
        "total_frames": int(total_frames),
    }


def main():
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    data_dir = out_dir / "data" / "chunk-000"
    meta_dir = out_dir / "meta"

    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    raw_episodes = sorted([p for p in raw_dir.glob("episode_*") if p.is_dir()])
    if not raw_episodes:
        raise FileNotFoundError(f"No episode_* folders found under {raw_dir}")

    episodes_rows = []
    episodes_stats_rows = []

    for episode_index, raw_episode_dir in enumerate(raw_episodes):
        print(f"Converting {raw_episode_dir.name} -> episode_{episode_index:06d}.parquet")
        result = convert_episode(
            raw_episode_dir=raw_episode_dir,
            out_chunk_dir=data_dir,
            episode_index=episode_index,
            fps=args.fps,
        )
        episodes_rows.append(result["episode_meta"])
        episodes_stats_rows.append(result["episode_stats"])

    info = {
        "dataset_name": out_dir.name,
        "robot_type": args.robot_type,
        "task_name": args.task_name,
        "fps": args.fps,
        "features": {
            "observation.image": "image_bytes",
            "observation.wrist_image": "image_bytes",
            "observation.state": "float_list",
            "action": "float_list",
            "joint_command": "float_list",
            "gripper_command": "float",
            "tcp_pos": "float_list",
            "joint_pos": "float_list",
            "joint_vel": "float_list",
            "gripper_pos": "float_list",
        },
    }

    tasks = [
        {
            "task_index": 0,
            "task_name": args.task_name,
        }
    ]

    stats = aggregate_global_stats(episodes_stats_rows)

    save_json(meta_dir / "info.json", info)
    save_json(meta_dir / "stats.json", stats)
    save_jsonl(meta_dir / "episodes.jsonl", episodes_rows)
    save_jsonl(meta_dir / "episodes_stats.jsonl", episodes_stats_rows)
    save_jsonl(meta_dir / "tasks.jsonl", tasks)

    print(f"Done. Converted dataset written to: {out_dir}")


if __name__ == "__main__":
    main()