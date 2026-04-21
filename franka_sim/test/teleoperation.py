import argparse
import time
from pathlib import Path
import json
import numpy as np

KEY_SPACE = 32


def parse_args():
    parser = argparse.ArgumentParser(description="Teleoperate FR3 simulation or use it as an IK controller.")
    parser.add_argument(
        "--mode",
        choices=("task", "controller"),
        default="task",
        help="Run a simulated task or a Franka-only controller scene.",
    )
    parser.add_argument("--task", choices=("gear",), default="gear", help="Task scene used when --mode task.")
    parser.add_argument(
        "--input",
        "--input-device",
        dest="input_device",
        choices=("xbox", "keyboard"),
        default="xbox",
        help="Input device used for teleoperation.",
    )
    parser.add_argument("--use-keyboard", action="store_true", help="Shortcut for --input keyboard.")
    parser.add_argument(
        "--env",
        choices=("gear", "franka"),
        default=None,
        help="Legacy shortcut: gear maps to --mode task, franka maps to --mode controller.",
    )
    parser.add_argument("--gui", action="store_true", help="Compatibility flag; the MuJoCo viewer is always used.")
    parser.add_argument("--position-step", type=float, default=0.01, help="Keyboard XYZ action step.")
    parser.add_argument("--rotation-step", type=float, default=0.05, help="Keyboard roll/pitch/yaw action step.")
    parser.add_argument("--gripper-step", type=float, default=0.1, help="Keyboard gripper action step.")
    parser.add_argument("--print-actions", action="store_true", help="Print raw input actions each step.")
    parser.add_argument("--print-joints", action="store_true", help="Print 7D arm joint commands each step.")

    parser.add_argument("--record", action="store_true", help="Record teleoperation data.")
    parser.add_argument("--dataset-dir", type=str, default="dataset_raw", help="Directory to save recorded episodes.")
    parser.add_argument("--save-images", action="store_true", help="Save front and wrist camera images.")

    parser.add_argument("--show-camera-view", action="store_true", help="Show live front/wrist camera view.")
    parser.add_argument("--camera-scale", type=float, default=2.0, help="Scale factor for live camera view.")


    return parser.parse_args()


def resolve_args(args):
    if args.use_keyboard:
        args.input_device = "keyboard"

    if args.env == "franka":
        args.mode = "controller"
    elif args.env == "gear":
        args.mode = "task"
        args.task = "gear"

    return args


def make_input_controller(args):
    if args.input_device == "keyboard":
        from keyboard_input import KeyboardInput

        return KeyboardInput(
            position_step=args.position_step,
            rotation_step=args.rotation_step,
            gripper_step=args.gripper_step,
        )

    from xbox_input import XboxInput

    return XboxInput()


def make_env(args):
    from franka_sim import envs

    if args.mode == "controller":
        return envs.FrankaTeleopGymEnv(action_scale=(0.1, 1))

    if args.task == "gear":
        return envs.PandaAssembleGearGymEnv(action_scale=(0.1, 1))

    raise ValueError(f"Unsupported task: {args.task}")


def maybe_print_step(args, action, info):
    if args.print_actions:
        print("action:", action)
    if args.print_joints and "joint_command" in info:
        print("joint_command:", info["joint_command"])


def save_image(path, image):
    import matplotlib.pyplot as plt

    plt.imsave(path, image)


def save_episode(episode_dir, episode_steps):
    if episode_dir is None:
        return

    steps_path = episode_dir / "steps.jsonl"
    with open(steps_path, "w", encoding="utf-8") as f:
        for step in episode_steps:
            f.write(json.dumps(step) + "\n")

    print(f"Saved episode to {episode_dir}")
    print(f"Saved {len(episode_steps)} steps.")


def make_episode_dirs(dataset_dir, episode_idx, save_images):
    episode_dir = dataset_dir / f"episode_{episode_idx:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    front_dir = None
    wrist_dir = None

    if save_images:
        front_dir = episode_dir / "front"
        wrist_dir = episode_dir / "wrist"
        front_dir.mkdir(exist_ok=True)
        wrist_dir.mkdir(exist_ok=True)

    return episode_dir, front_dir, wrist_dir


def finalize_and_start_new_episode(
    args,
    dataset_dir,
    episode_idx,
    episode_steps,
):
    if len(episode_steps) > 0:
        prev_episode_dir = dataset_dir / f"episode_{episode_idx:06d}"
        save_episode(prev_episode_dir, episode_steps)

    new_episode_idx = episode_idx + 1
    new_episode_dir, new_front_dir, new_wrist_dir = make_episode_dirs(
        dataset_dir, new_episode_idx, args.save_images
    )

    print(f"Started new episode: {new_episode_dir}")

    return new_episode_idx, [], 0, new_episode_dir, new_front_dir, new_wrist_dir


def update_viewer_overlay(viewer, obs, action=None):
    import mujoco
    from PIL import Image

    if "images" not in obs:
        try:
            viewer.clear_images()
            viewer.clear_texts()
        except Exception:
            pass
        return

    overlays = []

    def resize_img_2x(img):
        h, w = img.shape[:2]
        img_resized = Image.fromarray(img).resize((w * 2, h * 2), Image.NEAREST)
        arr = np.array(img_resized, dtype=np.uint8)
        arr = np.ascontiguousarray(arr)
        return arr

    # front
    if "front" in obs["images"]:
        img = obs["images"]["front"]
        print("front min/max:", img.min(), img.max())
        img_resized = resize_img_2x(img)
        h, w = img_resized.shape[:2]
        overlays.append((
            mujoco.MjrRect(620, 60, w, h),
            img_resized
        ))

    # wrist
    if "wrist" in obs["images"]:
        img = obs["images"]["wrist"]
        print("wrist min/max:", img.min(), img.max())
        img_resized = resize_img_2x(img)
        h, w = img_resized.shape[:2]
        overlays.append((
            mujoco.MjrRect(880, 60, w, h),
            img_resized
        ))

    try:
        viewer.set_images(overlays)
    except Exception as e:
        print("set_images failed:", e)

    if action is not None:
        try:
            viewer.set_texts((
                None,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "action",
                np.array2string(action, precision=3, suppress_small=True),
            ))
        except Exception as e:
            print("set_texts failed:", e)


def build_step_record(episode_idx, frame_idx, action, obs, info):
    return {
        "episode_index": episode_idx,
        "frame_id": frame_idx,
        "timestamp": time.time(),
        "action": action.tolist(),
        "joint_command": info["joint_command"].tolist() if "joint_command" in info else None,
        "gripper_command": float(info["gripper_command"]) if "gripper_command" in info else None,
        "tcp_pos": info["tcp_pos"].tolist() if "tcp_pos" in info else None,
        "joint_pos": obs["state"]["panda/joint_pos"].tolist(),
        "joint_vel": obs["state"]["panda/joint_vel"].tolist(),
        "gripper_pos": obs["state"]["panda/gripper_pos"].tolist(),
    }

def save_step_images(front_dir, wrist_dir, frame_idx, obs, step_record):
    if "images" not in obs:
        return

    if "front" in obs["images"] and front_dir is not None:
        front_path = front_dir / f"{frame_idx:06d}.png"
        save_image(front_path, obs["images"]["front"])
        step_record["front_path"] = f"front/{frame_idx:06d}.png"

    if "wrist" in obs["images"] and wrist_dir is not None:
        wrist_path = wrist_dir / f"{frame_idx:06d}.png"
        save_image(wrist_path, obs["images"]["wrist"])
        step_record["wrist_path"] = f"wrist/{frame_idx:06d}.png"

def main():
    args = resolve_args(parse_args())

    import mujoco.viewer

    episode_steps = []
    episode_idx = 0
    frame_idx = 0
    episode_dir = None
    front_dir = None
    wrist_dir = None
    dataset_dir = None

    if args.record:
        dataset_dir = Path(args.dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        episode_dir, front_dir, wrist_dir = make_episode_dirs(
            dataset_dir, episode_idx, args.save_images
        )

        print(f"Recording enabled. Episode dir: {episode_dir}")
        if args.save_images:
            print("Image saving enabled: front + wrist")
        print("Press reset to end current episode and start a new one.")

    env = make_env(args)
    model = env.model
    data = env.data
    reset_requested = False

    print(f"Teleoperation mode: {args.mode}, input: {args.input_device}")
    if args.mode == "controller":
        print("Controller mode returns joint commands in info['joint_command'].")

    def key_callback(keycode):
        nonlocal reset_requested
        if keycode == KEY_SPACE:
            reset_requested = True

    viewer_key_callback = None if args.input_device == "keyboard" else key_callback

    env.reset()
    controller = None

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=viewer_key_callback) as viewer:
            controller = make_input_controller(args)
            running = True

            while running and viewer.is_running():
                controller.poll_events()

                if controller.should_exit():
                    running = False
                    continue

                if hasattr(controller, "consume_reset_requested") and controller.consume_reset_requested():
                    reset_requested = True

                if reset_requested:
                    if args.record and dataset_dir is not None:
                        episode_idx, episode_steps, frame_idx, episode_dir, front_dir, wrist_dir = finalize_and_start_new_episode(
                            args=args,
                            dataset_dir=dataset_dir,
                            episode_idx=episode_idx,
                            episode_steps=episode_steps,
                        )

                    env.reset()
                    reset_requested = False
                    viewer.sync()
                    continue

                action = controller.get_action()
                step_start = time.time()

                obs, _, _, _, info = env.step(action)
                maybe_print_step(args, action, info)

                if args.show_camera_view:
                    live_obs = {"images": env.grab_images()}
                    update_viewer_overlay(viewer, live_obs, action=action)

                if args.record:
                    step_record = build_step_record(episode_idx, frame_idx, action, obs, info)

                    if args.save_images:
                        save_step_images(front_dir, wrist_dir, frame_idx, obs, step_record)

                    episode_steps.append(step_record)
                    frame_idx += 1

                    if frame_idx % 50 == 0:
                        print(f"Recorded {frame_idx} frames...")

                viewer.sync()

                time_until_next_step = env.control_dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    finally:
        if args.record and episode_dir is not None and len(episode_steps) > 0:
            save_episode(episode_dir, episode_steps)

        if args.show_camera_view:
            try:
                viewer.clear_images()
                viewer.clear_texts()
            except Exception:
                pass

        if controller is not None:
            controller.close()


if __name__ == "__main__":
    main()