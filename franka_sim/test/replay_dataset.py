import argparse
import io
import time

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
from PIL import Image

from franka_sim.envs.franka_teleop_env import FrankaTeleopGymEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Replay recorded parquet episode with MuJoCo main render.")
    parser.add_argument(
        "--parquet",
        type=str,
        required=True,
        help="Path to episode_xxxxxx.parquet",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="Replay FPS",
    )
    return parser.parse_args()


def decode_image(cell):
    if cell is None:
        return None
    if isinstance(cell, dict) and "bytes" in cell:
        raw = cell["bytes"]
        return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    return None


def update_replay_overlay(viewer, ds_front, ds_wrist, action=None):
    overlays = []

    def resize_img_2x(img):
        return np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)

    # recorded front
    if ds_front is not None:
        img = resize_img_2x(ds_front)
        h, w = img.shape[:2]
        overlays.append((mujoco.MjrRect(620, 60, w, h), img))

    # recorded wrist
    if ds_wrist is not None:
        img = resize_img_2x(ds_wrist)
        h, w = img.shape[:2]
        overlays.append((mujoco.MjrRect(880, 60, w, h), img))

    try:
        viewer.set_images(overlays)
    except Exception:
        pass

    if action is not None:
        try:
            viewer.set_texts((
                None,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "recorded action",
                np.array2string(np.asarray(action), precision=3, suppress_small=True),
            ))
        except Exception:
            pass


def main():
    args = parse_args()

    df = pd.read_parquet(args.parquet)

    env = FrankaTeleopGymEnv(image_obs=True, render_mode="rgb_array")
    model = env.model
    data = env.data
    env.reset()

    dt = 1.0 / args.fps

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for _, row in df.iterrows():
            step_start = time.time()

            action = np.array(row["action"], dtype=np.float32)

            # 保留主场景重放
            env.step(action)

            # 只显示录制结果
            ds_front = decode_image(row["observation.image"]) if "observation.image" in row else None
            ds_wrist = decode_image(row["observation.wrist_image"]) if "observation.wrist_image" in row else None

            update_replay_overlay(viewer, ds_front, ds_wrist, action=action)
            viewer.sync()

            sleep_time = dt - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        try:
            viewer.clear_images()
            viewer.clear_texts()
        except Exception:
            pass


if __name__ == "__main__":
    main()