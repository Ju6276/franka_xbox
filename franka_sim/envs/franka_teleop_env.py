from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import gym
import mujoco
import numpy as np
from gym import spaces

from franka_sim.controllers import opspace
from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "franka_only.xml"
_PANDA_HOME = np.asarray((0, -0.405, 0, -2.86, 0, 2.43, np.pi / 4)) # 机器人初始关节位置
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0.05], [0.7, 0.3, 0.7]]) # TCP位置的边界 [min_x, min_y, min_z], [max_x, max_y, max_z]
_TCP_POS_SENSOR = "panda/pinch_pos"  # MuJoCo传感器名称，用于获取TCP位置
_TCP_QUAT_SENSOR = "panda/pinch_quat"
_TCP_VEL_SENSOR = "panda/pinch_vel"


class FrankaTeleopGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.01, 1, 0.01]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = float("inf"),
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = True,
    ):
        if len(action_scale) < 3:
            action_scale = np.asarray([action_scale[0], action_scale[1], 0.1])
        self._action_scale = action_scale

        super().__init__(                   #采用MujocoGymEnv的构造函数，传入xml路径、随机种子、控制时间步长、物理时间步长、时间限制、渲染规格等参数
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }                                                                      #设置环境的元数据，包括支持的渲染模式和渲染帧率（根据控制时间步长计算）
        self.render_mode = render_mode
        self.image_obs = image_obs

        self._panda_dof_ids = np.asarray([self._model.joint(f"joint{i}").id for i in range(1, 8)])
        self._panda_qpos_ids = self._model.jnt_qposadr[self._panda_dof_ids]
        self._panda_qvel_ids = self._model.jnt_dofadr[self._panda_dof_ids]
        self._panda_ctrl_ids = np.asarray([self._model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._finger_joint_ids = np.asarray([self._model.joint(f"finger_joint{i}").id for i in range(1, 3)])
        self._finger_qpos_ids = self._model.jnt_qposadr[self._finger_joint_ids]
        self._finger_qvel_ids = self._model.jnt_dofadr[self._finger_joint_ids]
        self._finger_joint_ranges = self._model.jnt_range[self._finger_joint_ids]
        self._pinch_site_id = self._model.site("pinch").id

        observation_spaces = {
            "state": gym.spaces.Dict(
                {
                    "panda/tcp_pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                    "panda/tcp_vel": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                    "panda/joint_pos": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                    "panda/joint_vel": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                    "panda/gripper_pos": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                }
            )
        }
        if self.image_obs:
            observation_spaces["images"] = gym.spaces.Dict(
                {
                    "front": gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=(render_spec.height, render_spec.width, 3),
                        dtype=np.uint8,
                    ),
                    "wrist": gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=(render_spec.height, render_spec.width, 3),
                        dtype=np.uint8,
                    ),
                }
            )
        self.observation_space = gym.spaces.Dict(observation_spaces)

        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        self._rgb_renderer = mujoco.Renderer(
            self._model,
            height=render_spec.height,
            width=render_spec.width,
        )                                     

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[self._panda_qpos_ids] = _PANDA_HOME
        self._reset_gripper(open_gripper=True)
        mujoco.mj_forward(self._model, self._data)

        self._data.mocap_pos[0] = self._data.sensor(_TCP_POS_SENSOR).data
        self._data.mocap_quat[0] = self._data.sensor(_TCP_QUAT_SENSOR).data
        mujoco.mj_forward(self._model, self._data)

        obs = self._compute_observation()
        return obs, self._command_info()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        x, y, z, roll, pitch, yaw, grasp = action

        pos = self._data.mocap_pos[0].copy()                                                    #获取当前TCP位置的副本
        dpos = np.asarray([x, y, z]) * self._action_scale[0]                                     #根据动作的前3个元素（位置增量）和位置缩放因子计算TCP位置的增量
        self._data.mocap_pos[0] = np.clip(pos + dpos, *_CARTESIAN_BOUNDS)                       #将新的TCP位置限制在预定义的笛卡尔空间边界内，并更新mocap_pos[0]以反映新的TCP位置

        if grasp <= -1.0:
            self._open_gripper()
        elif grasp >= 1.0:
            self._set_gripper_command(open_gripper=False)
        else:
            g = self._data.ctrl[self._gripper_ctrl_id] / 255
            dg = grasp * self._action_scale[1]
            ng = np.clip(g + dg, 0.0, 1.0)
            self._data.ctrl[self._gripper_ctrl_id] = ng * 255

        current_quat = self._data.mocap_quat[0].copy()
        euler_increment = np.asarray([roll, pitch, yaw]) * self._action_scale[2]
        self._data.mocap_quat[0] = self.quat_multiply(current_quat, self.euler_to_quat(euler_increment))

        for _ in range(self._n_substeps):                                                      #给定目标 TCP 位姿，opspace(...) 根据当前状态和目标计算关节力矩 tau，把 tau 写进机械臂 7 个 actuator，MuJoCo 做一步物理仿真
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=_PANDA_HOME,
                gravity_comp=True,
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        terminated = self.time_limit_exceeded()
        return obs, 0.0, terminated, False, self._command_info()             #返回新的观察、奖励（这里始终为0.0，因为是遥操作不是强化学习）、是否终止（根据时间限制）、是否截断（这里始终为False）以及额外信息（当前命令信息）

    def render_camera(self, camera_name: str) -> np.ndarray:
        camera_id = mujoco.mj_name2id(
            self._model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
        )
        if camera_id == -1:
            raise ValueError(f"Camera '{camera_name}' not found in MuJoCo model.")

        self._rgb_renderer.update_scene(self._data, camera=camera_id)
        img = self._rgb_renderer.render()
        return np.ascontiguousarray(img, dtype=np.uint8)
    

    def grab_images(self) -> Dict[str, np.ndarray]:
        images = {
            "front": self.render_camera("front"),
            "wrist": self.render_camera("handcam_rgb"),
        }
        return images

    def render(self):
        return self.render_camera("front")

    def get_joint_command(self) -> np.ndarray:
        return self._data.qpos[self._panda_qpos_ids].astype(np.float32).copy()

    def get_gripper_command(self) -> float:
        return float(self._data.ctrl[self._gripper_ctrl_id] / 255)

    def _command_info(self) -> Dict[str, Any]:  
        return {
            "joint_command": self.get_joint_command(),
            "gripper_command": self.get_gripper_command(),
            "tcp_pos": self._data.sensor(_TCP_POS_SENSOR).data.astype(np.float32).copy(),
        }

    def _set_gripper_command(self, open_gripper: bool):
        self._data.ctrl[self._gripper_ctrl_id] = 0.0 if open_gripper else 255.0

    def _open_gripper(self):
        self._reset_gripper(open_gripper=True)

    def _reset_gripper(self, open_gripper: bool):
        target_qpos = self._finger_joint_ranges[:, 1] if open_gripper else self._finger_joint_ranges[:, 0]
        self._data.qpos[self._finger_qpos_ids] = target_qpos
        self._data.qvel[self._finger_qvel_ids] = 0.0
        self._set_gripper_command(open_gripper=open_gripper)
        mujoco.mj_forward(self._model, self._data)

    def get_state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "panda/tcp_pos": self._data.sensor(_TCP_POS_SENSOR).data.astype(np.float32),
            "panda/tcp_vel": self._data.sensor(_TCP_VEL_SENSOR).data.astype(np.float32),
            "panda/joint_pos": self._data.qpos[self._panda_qpos_ids].astype(np.float32),
            "panda/joint_vel": self._data.qvel[self._panda_qvel_ids].astype(np.float32),
            "panda/gripper_pos": np.array([self.get_gripper_command()], dtype=np.float32),
        }

    def _compute_observation(self) -> Dict[str, Any]:       #从 MuJoCo 模型中提取当前状态信息，构建一个包含 TCP 位置、TCP 速度、关节位置、关节速度和夹爪位置的观察字典，如果启用了图像观察，还会包含前视和手腕摄像头的图像数据
        obs = {
                "state": self.get_state_dict()
        }
        if self.image_obs:
            obs["images"] = self.grab_images()
        return obs

    def euler_to_quat(self, euler):
        roll, pitch, yaw = euler

        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([x, y, z, w])

    def quat_multiply(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([x, y, z, w])