# franka_bridge.py

from __future__ import annotations

import threading
import time
from typing import Optional, Sequence

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.action import ActionClient
from franka_msgs.action import Move

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class FrankaBridge:
    """
    ROS 2 bridge for Franka Emika Panda through pixi_panda_ros2.

    Expected active controller:
        joint_trajectory_controller

    Expected command topic:
        /joint_trajectory_controller/joint_trajectory

    Expected joint names:
        panda_joint1 ... panda_joint7
    """

    def __init__(
        self,
        robot_ip: Optional[str],
        home_q: Sequence[float],
        command_rate_hz: float = 100.0,
        max_joint_step: float = 0.002,
        gripper_speed: float = 0.04,
        limit_rate: bool = True,
        cutoff_frequency: Optional[float] = 100.0,
        arm_controller: str = "/joint_trajectory_controller",
        joint_state_topic: str = "/joint_states",
        joint_names: Optional[Sequence[str]] = None,
    ):
        self.robot_ip = robot_ip
        self.home_q = np.asarray(home_q, dtype=np.float64)

        if self.home_q.shape != (7,):
            raise ValueError(f"home_q must have shape (7,), got {self.home_q.shape}")

        self.command_rate_hz = float(command_rate_hz)
        self.command_period = 1.0 / self.command_rate_hz
        self.max_joint_step = float(max_joint_step)
        self.gripper_speed = float(gripper_speed)
        self.limit_rate = bool(limit_rate)
        self.cutoff_frequency = cutoff_frequency

        self.arm_controller = arm_controller.rstrip("/")
        self.joint_trajectory_topic = f"{self.arm_controller}/joint_trajectory"

        self.joint_names = list(
            joint_names
            if joint_names is not None
            else [f"panda_joint{i}" for i in range(1, 8)]
        )

        self._last_q = self.home_q.copy()
        self._latest_robot_q: Optional[np.ndarray] = None
        self._last_publish_time = time.monotonic()

        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()

        self.node = Node("panda_bridge")

        # Gripper action client setup
        self.gripper_action_name = "/panda_gripper/move"
        self.gripper_speed = float(gripper_speed)

        self.gripper_client = ActionClient(
            self.node,
            Move,
            self.gripper_action_name,
        )

        if not self.gripper_client.wait_for_server(timeout_sec=2.0):
            self.node.get_logger().warning(
                f"Gripper action server not available: {self.gripper_action_name}"
            )
            self.gripper_client = None
        else:
            self.node.get_logger().info(
                f"Connected to gripper action: {self.gripper_action_name}"
            )

        self._last_gripper_width = None
        self._last_gripper_send_time = 0.0
        self._gripper_goal_active = False

        self.gripper_min_interval = 0.5
        self.gripper_min_change = 0.005

        #publisher for sending joint trajectory commands
        self.joint_pub = self.node.create_publisher(    
            JointTrajectory,
            self.joint_trajectory_topic,
            10,
        )

        #subscriber for receiving joint state updates
        self.joint_state_sub = self.node.create_subscription(
            JointState,
            joint_state_topic,
            self._joint_state_callback,
            10,
        )

        #executor for running the ROS 2 node in a separate thread
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        self.spin_thread = threading.Thread(
            target=self.executor.spin,
            daemon=True,
        )
        self.spin_thread.start()

        self.node.get_logger().info(
            f"Panda FrankaBridge started. Publishing to {self.joint_trajectory_topic}"
        )

        if robot_ip is not None:
            self.node.get_logger().info(
                "robot_ip is kept for compatibility only. "
                "pixi_panda_ros2 handles the actual robot connection."
            )

        self._wait_for_joint_state(timeout=2.0)

        if self._latest_robot_q is not None:
            self._last_q = self._latest_robot_q.copy()
            self.node.get_logger().info(
                f"Initial robot q: {np.round(self._last_q, 4)}"
            )
        else:
            self.node.get_logger().warning(
                "No matching /joint_states received. Using home_q as initial q."
            )

    # Callback for processing incoming joint state messages
    def _joint_state_callback(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))

        if all(name in name_to_pos for name in self.joint_names):
            self._latest_robot_q = np.asarray(
                [name_to_pos[name] for name in self.joint_names],
                dtype=np.float64,
            )

    def _gripper_goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._gripper_goal_active = False
            self.node.get_logger().warning(f"Failed to send gripper goal: {exc}")
            return

        if not goal_handle.accepted:
            self._gripper_goal_active = False
            self.node.get_logger().warning("Gripper goal rejected.")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._gripper_result_callback)


    def _gripper_result_callback(self, future):
        self._gripper_goal_active = False

        try:
            result = future.result().result

            if hasattr(result, "success") and not result.success:
                error = getattr(result, "error", "")
                self.node.get_logger().warning(f"Gripper move failed: {error}")

        except Exception as exc:
            self.node.get_logger().warning(f"Failed to get gripper result: {exc}")

    def _wait_for_joint_state(self, timeout: float):
        deadline = time.monotonic() + timeout

        while self._latest_robot_q is None and time.monotonic() < deadline:
            time.sleep(0.01)

    def _check_joint_command(self, q: Sequence[float]) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)

        if q.shape != (7,):
            raise ValueError(f"joint command must have shape (7,), got {q.shape}")

        if not np.all(np.isfinite(q)):
            raise ValueError(f"joint command contains NaN or Inf: {q}")

        return q

    def _lowpass(self, target_q: np.ndarray, dt: float) -> np.ndarray:
        if self.cutoff_frequency is None or self.cutoff_frequency <= 0.0:
            return target_q

        rc = 1.0 / (2.0 * np.pi * self.cutoff_frequency)
        alpha = dt / (rc + dt)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        return self._last_q + alpha * (target_q - self._last_q)

    #builds and publishes a JointTrajectory message 
    def _publish_joint_trajectory(self, q: np.ndarray, duration: float):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = q.tolist()
        point.time_from_start = Duration(seconds=float(duration)).to_msg()

        msg.points = [point]
        self.joint_pub.publish(msg)

    def move_to_home(self, duration: float = 4.0):
        q_home = self._check_joint_command(self.home_q)

        self.node.get_logger().info(
            f"Moving to home over {duration:.2f} s: {np.round(q_home, 4)}"
        )

        self._publish_joint_trajectory(q_home, duration=duration)

        time.sleep(max(0.0, duration))

        self._last_q = q_home.copy()
        self._last_publish_time = time.monotonic()

    # Main method for updating joint targets. Teleop code should call this at a high rate (e.g. 100 Hz) with the latest desired joint positions.
    def update_joint_target(self, joint_command: Sequence[float]):
        target_q = self._check_joint_command(joint_command)

        now = time.monotonic()
        dt = max(now - self._last_publish_time, self.command_period)

        #limit the rate of change of joint commands to prevent sending overly aggressive trajectories that may cause pixi_panda_ros2 to reject them
        if self.limit_rate:
            delta = target_q - self._last_q
            delta = np.clip(delta, -self.max_joint_step, self.max_joint_step)
            target_q = self._last_q + delta

        #lowpass filter 
        target_q = self._lowpass(target_q, dt)

        #compute trajectory duration based on command rate, with a minimum of 50 ms to ensure the controller accepts it
        horizon = max(3.0 * self.command_period, 0.05)

        self._publish_joint_trajectory(target_q, duration=horizon)

        self._last_q = target_q.copy()
        self._last_publish_time = now

    # Main method for updating gripper targets. Teleop code should call this with the latest desired gripper width.
    def update_gripper_target(self, width: float):
        """
        Send Panda gripper width command through /panda_gripper/move.

        width:
            0.08 = fully open
            0.00 = fully closed
        """
        width = float(np.clip(width, 0.0, 0.08))

        if self.gripper_client is None:
            return

        now = time.monotonic()

        # First call only records state; do not move gripper at teleop startup.
        if self._last_gripper_width is None:
            self._last_gripper_width = width
            return

        # Avoid sending tiny changes.
        if abs(width - self._last_gripper_width) < self.gripper_min_change:
            return

        # Avoid spamming gripper action.
        if now - self._last_gripper_send_time < self.gripper_min_interval:
            return

        # Avoid sending a new gripper goal while the previous one is still running.
        if self._gripper_goal_active:
            return

        goal = Move.Goal()
        goal.width = width
        goal.speed = self.gripper_speed

        future = self.gripper_client.send_goal_async(goal)
        future.add_done_callback(self._gripper_goal_response_callback)

        self._gripper_goal_active = True
        self._last_gripper_width = width
        self._last_gripper_send_time = now

    def close(self):
        try:
            self.node.get_logger().info("Closing Panda FrankaBridge.")
        except Exception:
            pass

        try:
            self.executor.shutdown()
        except Exception:
            pass

        try:
            self.node.destroy_node()
        except Exception:
            pass

        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()