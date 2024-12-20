from typing import Any, List, Union, Tuple
import genesis as gs
from genesis.engine.entities import RigidEntity
import gymnasium as gym
import numpy as np
import torch
class BaseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    def __init__(self,
            num_envs: int,
            sim_options: gs.options.SimOptions = None,
            viewer_options: gs.options.ViewerOptions = None,
            rigid_options: gs.options.RigidOptions = None,
            control_freq: int = 50,
            control_mode: str = "pd_joint_delta_pos",
            render_mode: str = "rgb_array",
            env_spacing: Tuple[float, float] = (5.0, 5.0),
            robot_uids: Union[str, List[str]] = "panda"
        ):


        # substeps = 4 is necessary for grasp capabilities
        self.scene = gs.Scene(
            show_viewer    = self.render_mode == "human",
            sim_options = sim_options if sim_options is not None else gs.options.SimOptions(
                dt = 0.01,
                substeps = 4 # related to solver iterations essentially?,
            ),
            viewer_options = viewer_options if viewer_options is not None else gs.options.ViewerOptions(
                camera_pos    = (0.5, -0.5, 0.5),
                camera_lookat = (0.0, 0.0, 0.25),
                camera_fov    = 40,
                max_FPS = 60,
            ),
            rigid_options = rigid_options if rigid_options is not None else gs.options.RigidOptions()
        )
        self.control_mode = control_mode
        self.sim_freq = int(1 / sim_options.dt)
        self.control_freq = control_freq
        self.sim_steps_per_control_step = self.sim_freq // control_freq
        assert self.sim_freq % self.control_freq == 0, "sim_freq must be divisible by control_freq"
        self.num_envs = num_envs
        self.render_mode = render_mode

        self._load_scene()
        self._load_sensors()
        self.scene.build(n_envs=self.num_envs, env_spacing=env_spacing)

        if robot_uids == "panda":
            self.rest_qpos = torch.tensor(
            [
                0.0,
                np.pi / 8,
                0,
                -np.pi * 5 / 8,
                0,
                np.pi * 3 / 4,
                np.pi / 4,
                0.04,
                0.04,
            ], device=gs.device)
            self.motor_dofs = torch.arange(9, device=gs.device)
            self.arm_dofs = self.motor_dofs[:7]
            self.gripper_dofs = self.motor_dofs[7:]
            # NOTE (stao): it is difficult to match the stiffness/damping parameters here with physx based sims, so genesis will behave differently
            # self.robot.set_dofs_kp(
            #     np.array([1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3]),
            # )
            # self.robot.set_dofs_kv(
            #     np.array([1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2]),
            # )

            # set stiffness/damping based on genesis tutorial docs
            self.robot.set_dofs_kp(
                kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
            )
            # set velocity gains
            self.robot.set_dofs_kv(
                kv             = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
            )
            self.qlimits = self.robot.get_dofs_limit()
        else:
            raise NotImplementedError(f"Robot {robot_uids} not supported in this benchmark")

        if self.control_mode == "pd_joint_delta_pos" or self.control_mode == "pd_joint_target_delta_pos":
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_envs, 9))
            self.single_action_space = gym.spaces.Box(low=-1, high=1, shape=(9, ))
        elif self.control_mode == "pd_joint_pos":
            self.action_space = gym.spaces.Box(low=self.qlimits[0].cpu().numpy(), high=self.qlimits[1].cpu().numpy(), shape=(self.num_envs, 9))
            self.single_action_space = gym.spaces.Box(low=self.qlimits[0].cpu().numpy(), high=self.qlimits[1].cpu().numpy(), shape=(9, ))

        self.fixed_trajectory = {}

        self._target_qpos = torch.zeros((self.num_envs, 9), device=gs.device)
    def _load_scene(self):
        raise NotImplementedError()
    def _load_sensors(self):
        raise NotImplementedError()
    def set_control_mode(self, control_mode: str):
        self.control_mode = control_mode

    def get_obs(self):
        raise NotImplementedError()
    def step(self, action):
        if self.control_mode == "pd_joint_delta_pos":
            action = torch.clip(action, -1, 1)
            # match maniskill action scaling
            arm_action = action[:, :7] * 0.1
            gripper_action = action[:, 7:] * 0.05 - 0.01
            self._target_qpos = torch.cat([arm_action, gripper_action], dim=-1) + self.robot.get_dofs_position(self.motor_dofs)

            # below implementation probably has some marginal overhead compared to above. But below is closer to maniskill which does it this way
            # for more readability and flexibility in controller design
            # self.robot.control_dofs_position(arm_action + self.robot.get_dofs_position(self.arm_dofs), self.arm_dofs)
            # self.robot.control_dofs_position(gripper_action + self.robot.get_dofs_position(self.gripper_dofs), self.gripper_dofs)
        elif self.control_mode == "pd_joint_target_delta_pos":
            action = torch.clip(action, -1, 1)
            # match maniskill action scaling
            arm_action = action[:, :7] * 0.02 # lower to account for the fact genesis covers more distance than usual
            gripper_action = action[:, 7:] * 0.05 - 0.01
            self._target_qpos = torch.cat([arm_action, gripper_action], dim=-1) + self._target_qpos
        elif self.control_mode == "pd_joint_pos":
            target_qpos = action
            self._target_qpos = target_qpos
        self.robot.control_dofs_position(self._target_qpos)

        # TODO (stao): is it better to change dt or substeps, or is this the correct way to do more sim steps per control step?
        for _ in range(self.sim_steps_per_control_step):
            self.scene.step()
        return self.get_obs(), None, None, None, {}
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self._initialize_episode()
        self._target_qpos = self.robot.get_dofs_position(self.motor_dofs)
        return self.get_obs(), {}
    def _initialize_episode(self):
        raise NotImplementedError()
    def render_rgb_array(self):
        rgb = self.cam.render(rgb=True)[0]
        return rgb
