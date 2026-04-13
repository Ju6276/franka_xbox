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
        default="keyboard",
        help="Input device used for teleoperation.",
    )
    parser.add_argument("--xml", type=Path, default=XML_PATH, help="MuJoCo XML used for kinematic IK.")
    parser.add_argument("--solver", default="daqp", help="QP solver used by mink.solve_ik.")
    parser.add_argument("--rate", type=float, default=100.0, help="Controller loop rate in Hz.")
    parser.add_argument("--no-viewer", action="store_true", help="Run without the MuJoCo viewer.")
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

    print("Standalone Mink controller is running.")
    print("This script prints joint commands only; it does not send commands to the real robot.")
    print("Use the command output as the place to connect your Franka bridge after safety checks.")

    def key_callback(keycode):
        nonlocal reset_requested
        if keycode == KEY_SPACE:
            reset_requested = True

    def teleop_loop(viewer=None):
        nonlocal gripper_command, reset_requested, step_count

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
                reset_requested = False

            action = controller.get_action()

            data.mocap_pos[target_mocap_id] = np.clip(
                data.mocap_pos[target_mocap_id] + action[:3],
                cartesian_bounds[0],
                cartesian_bounds[1],
            )
            delta_quat = euler_to_quat_wxyz(action[3:6])
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
