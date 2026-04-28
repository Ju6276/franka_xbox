import argparse
import math
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
XML_PATH = HERE.parent / "envs" / "xmls" / "franka_only.xml"

KEY_SPACE = 32
PANDA_HOME = (0.0, -0.405, 0.0, -2.86, 0.0, 2.43, math.pi / 4)
CARTESIAN_BOUNDS = ((0.2, -0.3, 0.02), (0.6, 0.3, 0.6))
np = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone Mink teleoperation controller for real Franka command generation."
    )
    parser.add_argument(
        "--input",
        "--input-device",
        dest="input_device",
        choices=("keyboard", "xbox"),
        default="xbox",
        help="Input device used for teleoperation.",
    )
    parser.add_argument("--xml", type=Path, default=XML_PATH, help="MuJoCo XML used for kinematic IK.")
    parser.add_argument("--solver", default="daqp", help="QP solver used by mink.solve_ik.")
    parser.add_argument("--rate", type=float, default=50.0, help="Controller loop rate in Hz.")
    parser.add_argument(
        "--no-viewer",
        dest="no_viewer",
        action="store_true",
        default=True,
        help="Run without the MuJoCo viewer.",
    )
    parser.add_argument(
        "--viewer",
        dest="no_viewer",
        action="store_false",
        help="Show the MuJoCo viewer.",
    )
    parser.add_argument("--position-step", type=float, default=0.002, help="Keyboard XYZ step in meters.")
    parser.add_argument("--rotation-step", type=float, default=0.02, help="Keyboard RPY step in radians.")
    parser.add_argument("--gripper-step", type=float, default=0.1, help="Incremental gripper command step.")
    parser.add_argument("--position-cost", type=float, default=1.0, help="Mink frame position task cost.")
    parser.add_argument("--orientation-cost", type=float, default=1.0, help="Mink frame orientation task cost.")
    parser.add_argument("--posture-cost", type=float, default=1e-3, help="Mink posture task cost.")
    parser.add_argument("--lm-damping", type=float, default=1e-4, help="Mink Levenberg-Marquardt damping.")
    parser.add_argument(
        "--max-joint-velocity",
        type=float,
        default=1.0,
        help="Joint velocity limit in rad/s for joint1-joint7.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print joint commands every N steps. Use 0 to disable printing.",
    )
    parser.add_argument("--xbox-position-scale", type=float, default=0.015)
    parser.add_argument("--xbox-rotation-scale", type=float, default=0.015)
    parser.add_argument(
        "--debug-action",
        dest="debug_action",
        action="store_true",
        default=True,
        help="Print action, dpos and drpy debug info.",
    )
    parser.add_argument(
        "--no-debug-action",
        dest="debug_action",
        action="store_false",
        help="Disable action debug printing.",
    )
    parser.add_argument(
        "--real",
        dest="real",
        action="store_true",
        default=True,
        help="Send commands to the real Franka using panda-py.",
    )
    parser.add_argument(
        "--sim",
        dest="real",
        action="store_false",
        help="Run without sending commands to the real robot.",
    )
    parser.add_argument("--robot-ip", default="192.168.3.100")
    parser.add_argument(
        "--move-home",
        dest="move_home",
        action="store_true",
        default=True,
        help="Move real robot to PANDA_HOME before starting.",
    )
    parser.add_argument(
        "--no-move-home",
        dest="move_home",
        action="store_false",
        help="Do not move real robot to PANDA_HOME before starting.",
    )
    parser.add_argument("--home-duration", type=float, default=20.0)
    parser.add_argument("--home-tol", type=float, default=0.03)

    parser.add_argument("--max-delta", type=float, default=0.15)
    parser.add_argument("--dq-alpha", type=float, default=0.08)
    parser.add_argument("--dq-rate-limit", type=float, default=0.003)
    parser.add_argument("--send-deadband", type=float, default=0.0)

    parser.add_argument(
        "--joint-mask",
        type=int,
        nargs=7,
        default=[1, 1, 1, 1, 1, 1, 1],
        help="Joint mask for real robot command.",
    )
    return parser.parse_args()


def quat_normalize(quat):
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return np.asarray((1.0, 0.0, 0.0, 0.0))
    return quat / norm


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.asarray(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def euler_to_quat_wxyz(euler):
    roll, pitch, yaw = euler
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    return quat_normalize(
        np.asarray(
            (
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            )
        )
    )


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


def set_site_mocap_pose(model, data, mocap_name, site_name):
    import mujoco

    mocap_id = int(model.body(mocap_name).mocapid[0])
    site_id = model.site(site_name).id
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
    data.mocap_pos[mocap_id] = data.site_xpos[site_id]
    data.mocap_quat[mocap_id] = quat_normalize(quat)
    return mocap_id


def set_visual_gripper(model, data, finger_qpos_ids, finger_qvel_ids, finger_ranges, gripper_command):
    close_qpos = finger_ranges[:, 0]
    open_qpos = finger_ranges[:, 1]
    target_qpos = open_qpos + gripper_command * (close_qpos - open_qpos)
    data.qpos[finger_qpos_ids] = target_qpos
    data.qvel[finger_qvel_ids] = 0.0
    data.ctrl[model.actuator("fingers_actuator").id] = gripper_command * 255.0


def update_gripper_command(gripper_command, grasp_action, gripper_step):
    if grasp_action <= -1.0:
        return 0.0
    if grasp_action >= 1.0:
        return 1.0
    return float(np.clip(gripper_command + grasp_action * gripper_step, 0.0, 1.0))


def gripper_width_from_command(gripper_command):
    return 0.08 * (1.0 - gripper_command)


def print_command(step_count, joint_command, gripper_command):
    joint_str = np.array2string(joint_command, precision=5, suppress_small=True)
    width = gripper_width_from_command(gripper_command)
    print(f"joint_command={joint_str} gripper_width={width:.4f} gripper_command={gripper_command:.3f}")


class PandaPyJointSender:
    def __init__(self, robot_ip):
        from panda_py import Panda
        from panda_py.controllers import JointPosition

        self.panda = Panda(robot_ip)
        self.panda.set_default_behavior()

        self.controller = JointPosition(
            stiffness=np.array([300, 300, 300, 300, 150, 100, 50], dtype=np.float64),
            damping=np.array([30, 30, 30, 20, 15, 10, 8], dtype=np.float64),
            filter_coeff=0.02,
        )

        self.panda.start_controller(self.controller)
        print("panda-py JointPosition controller started.")

    def read_q(self):
        return np.asarray(self.panda.q, dtype=np.float64)

    def send_q(self, q_target):
        self.controller.set_control(np.asarray(q_target, dtype=np.float64), np.zeros(7))
        self.panda.raise_error()

    def move_to(self, q_goal, duration=20.0):
        q_start = self.read_q()
        q_goal = np.asarray(q_goal, dtype=np.float64)

        print("Moving real robot to PANDA_HOME.")
        print("q_start =", np.round(q_start, 6))
        print("q_goal  =", np.round(q_goal, 6))
        print("delta   =", np.round(q_goal - q_start, 6))
        input("WARNING: real robot will move. Press Enter to continue...")

        dt = 0.02
        steps = max(1, int(duration / dt))

        for i in range(steps + 1):
            s = i / steps
            alpha = 0.5 - 0.5 * np.cos(np.pi * s)
            q = q_start + alpha * (q_goal - q_start)
            self.send_q(q)
            time.sleep(dt)

        print("Move to PANDA_HOME finished.")

    def stop(self):
        try:
            self.panda.stop_controller()
        except Exception:
            pass


def run(args):
    global np

    if np is None:
        import numpy as np_runtime

        np = np_runtime

    import mink
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    configuration = mink.Configuration(model)
    data = configuration.data
    panda_home = np.asarray(PANDA_HOME)
    real_sender = None

    if args.real:
        print("Connecting to real Franka with panda-py...")
        real_sender = PandaPyJointSender(args.robot_ip)

        q_real = real_sender.read_q()
        home_err = float(np.max(np.abs(q_real - panda_home)))

        print("real q     =", np.round(q_real, 6))
        print("PANDA_HOME =", np.round(panda_home, 6))
        print("home_err   =", home_err)

        if home_err > args.home_tol:
            if args.move_home:
                real_sender.move_to(panda_home, duration=args.home_duration)
            else:
                raise RuntimeError(
                    f"Real robot is not at PANDA_HOME. home_err={home_err:.6f}. "
                    f"Use --move-home if workspace is clear."
                )
    cartesian_bounds = np.asarray(CARTESIAN_BOUNDS)

    arm_joint_names = [f"joint{i}" for i in range(1, 8)]
    arm_joint_ids = np.asarray([model.joint(name).id for name in arm_joint_names])
    arm_qpos_ids = model.jnt_qposadr[arm_joint_ids]
    finger_joint_ids = np.asarray([model.joint(f"finger_joint{i}").id for i in range(1, 3)])
    finger_qpos_ids = model.jnt_qposadr[finger_joint_ids]
    finger_qvel_ids = model.jnt_dofadr[finger_joint_ids]
    finger_ranges = model.jnt_range[finger_joint_ids]

    q = data.qpos.copy()
    q[arm_qpos_ids] = panda_home
    q[finger_qpos_ids] = finger_ranges[:, 1]
    configuration.update(q)
    gripper_command = 0.0
    set_visual_gripper(model, data, finger_qpos_ids, finger_qvel_ids, finger_ranges, gripper_command)
    mujoco.mj_forward(model, data)

    target_mocap_id = set_site_mocap_pose(model, data, "target", "pinch")

    end_effector_task = mink.FrameTask(
        frame_name="pinch",
        frame_type="site",
        position_cost=args.position_cost,
        orientation_cost=args.orientation_cost,
        lm_damping=args.lm_damping,
    )
    posture_task = mink.PostureTask(model, cost=args.posture_cost)
    posture_task.set_target(configuration.q)

    max_velocities = {joint_name: args.max_joint_velocity for joint_name in arm_joint_names}
    try:
        velocity_limit = mink.VelocityLimit(model=model, velocities=max_velocities)
    except TypeError:
        velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits = [mink.ConfigurationLimit(model=model), velocity_limit]

    controller = make_input_controller(args)
    dt = 1.0 / args.rate
    step_count = 0
    reset_requested = False
    dq_filtered = np.zeros(7, dtype=np.float64)
    dq_last_sent = np.zeros(7, dtype=np.float64)

    print("Standalone Mink controller is running.")
    if real_sender is None:
        print("This script prints joint commands only; it does not send commands to the real robot.")
        print("Use the command output as the place to connect your Franka bridge after safety checks.")
    else:
        print("Real robot streaming is enabled via panda-py.")

    def key_callback(keycode):
        nonlocal reset_requested
        if keycode == KEY_SPACE:
            reset_requested = True

    def teleop_loop(viewer=None):
        nonlocal gripper_command, reset_requested, step_count
        nonlocal dq_filtered, dq_last_sent

        while True:
            loop_start = time.time()
            if viewer is not None and not viewer.is_running():
                break

            controller.poll_events()
            if controller.should_exit():
                break
            if hasattr(controller, "consume_reset_requested") and controller.consume_reset_requested():
                reset_requested = True
            if reset_requested:
                q = data.qpos.copy()
                q[arm_qpos_ids] = panda_home
                q[finger_qpos_ids] = finger_ranges[:, 1]
                configuration.update(q)
                gripper_command = 0.0
                set_visual_gripper(model, data, finger_qpos_ids, finger_qvel_ids, finger_ranges, gripper_command)
                mujoco.mj_forward(model, data)
                set_site_mocap_pose(model, data, "target", "pinch")
                posture_task.set_target(configuration.q)
                dq_filtered[:] = 0.0
                dq_last_sent[:] = 0.0

                if real_sender is not None:
                    real_sender.send_q(panda_home)
                    print("Reset requested -> sent PANDA_HOME to real robot.")
                reset_requested = False

            action = controller.get_action()

            if args.input_device == "xbox":
                dpos = action[:3] * args.xbox_position_scale
                drpy = action[3:6] * args.xbox_rotation_scale
            else:
                dpos = action[:3]
                drpy = action[3:6]

            data.mocap_pos[target_mocap_id] = np.clip(
                data.mocap_pos[target_mocap_id] + dpos,
                cartesian_bounds[0],
                cartesian_bounds[1],
            )

            delta_quat = euler_to_quat_wxyz(drpy)

            if args.debug_action and step_count % max(1, args.print_every) == 0:
                print(
                    "action=", np.round(action, 5),
                    "dpos=", np.round(dpos, 6),
                    "drpy=", np.round(drpy, 6),
                )
            data.mocap_quat[target_mocap_id] = quat_normalize(
                quat_multiply(data.mocap_quat[target_mocap_id], delta_quat)
            )

            gripper_command = update_gripper_command(gripper_command, action[6], args.gripper_step)

            end_effector_task.set_target(mink.SE3.from_mocap_name(model, data, "target"))
            velocity = mink.solve_ik(
                configuration,
                [end_effector_task, posture_task],
                dt,
                args.solver,
                limits=limits,
            )
            configuration.integrate_inplace(velocity, dt)
            set_visual_gripper(model, data, finger_qpos_ids, finger_qvel_ids, finger_ranges, gripper_command)
            mujoco.mj_forward(model, data)

            joint_command = data.qpos[arm_qpos_ids].astype(np.float64).copy()
            if args.print_every > 0 and step_count % args.print_every == 0:
                print_command(step_count, joint_command, gripper_command)

            if real_sender is not None:
                dq_unclipped = joint_command - panda_home

                mask = np.asarray(args.joint_mask, dtype=np.float64)
                dq_unclipped = dq_unclipped * mask

                dq_unclipped[np.abs(dq_unclipped) < args.send_deadband] = 0.0

                saturated = np.abs(dq_unclipped) > args.max_delta
                dq_raw = np.clip(dq_unclipped, -args.max_delta, args.max_delta)

                if args.print_every > 0 and step_count % args.print_every == 0:
                    if np.any(saturated):
                        print(
                            "WARNING: dq saturated. joints=",
                            np.where(saturated)[0] + 1,
                            "dq_unclipped=",
                            np.round(dq_unclipped, 4),
                        )

                dq_filtered = (1.0 - args.dq_alpha) * dq_filtered + args.dq_alpha * dq_raw

                step = np.clip(
                    dq_filtered - dq_last_sent,
                    -args.dq_rate_limit,
                    args.dq_rate_limit,
                )

                dq_to_send = dq_last_sent + step
                dq_last_sent = dq_to_send.copy()

                q_target = panda_home + dq_to_send
                real_sender.send_q(q_target)

                if args.print_every > 0 and step_count % args.print_every == 0:
                    q_real = real_sender.read_q()
                    home_dist_real = float(np.max(np.abs(q_real - panda_home)))
                    home_dist_cmd = float(np.max(np.abs(q_target - panda_home)))

                    print(
                        "dq_raw=", np.round(dq_raw, 5),
                        "dq_send=", np.round(dq_to_send, 5),
                        "home_real=", round(home_dist_real, 6),
                        "home_cmd=", round(home_dist_cmd, 6),
                    )

            if viewer is not None:
                viewer.sync()

            step_count += 1
            sleep_time = dt - (time.time() - loop_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    try:
        if args.no_viewer:
            teleop_loop(viewer=None)
        else:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
                teleop_loop(viewer=viewer)
    finally:
        controller.close()
        if real_sender is not None:
            real_sender.stop()


def main():
    global np

    args = parse_args()

    try:
        import numpy as np_runtime
        import mujoco  # noqa: F401
        import mink  # noqa: F401
        if not args.no_viewer:
            import mujoco.viewer  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency for Mink teleoperation. Install it in your active env, for example:\n"
            "  pip install numpy mujoco mink daqp\n"
            f"Original import error: {exc}"
        ) from exc

    np = np_runtime
    run(args)


if __name__ == "__main__":
    main()
