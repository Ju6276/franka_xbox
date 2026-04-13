import argparse
import time

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


def main():
    args = resolve_args(parse_args())

    import mujoco.viewer

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
                    env.reset()
                    reset_requested = False
                    viewer.sync()
                    continue

                action = controller.get_action()
                step_start = time.time()
                _, _, _, _, info = env.step(action)
                maybe_print_step(args, action, info)
                viewer.sync()

                time_until_next_step = env.control_dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    finally:
        if controller is not None:
            controller.close()


if __name__ == "__main__":
    main()
