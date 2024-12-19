from typing import Any, List, Union
import genesis as gs
import gymnasium as gym
import numpy as np
import torch
class FrankaPickCubeBenchmarkEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    def __init__(self, num_envs: int, sim_freq: int, control_freq: int, render_mode: str, control_mode: str = "pd_joint_delta_pos", enable_self_collision: bool = True, robot_uids: Union[str, List[str]] = "panda"):
        self.control_mode = control_mode
        self.sim_freq = sim_freq
        self.control_freq = control_freq
        self.sim_steps_per_control_step = sim_freq // control_freq
        self.num_envs = num_envs
        self.render_mode = render_mode

        # substeps = 4 is necessary for grasp capabilities
        self.scene = gs.Scene(
            show_viewer    = self.render_mode == "human",
            sim_options = gs.options.SimOptions(
                dt = 1 / sim_freq,
                substeps = 4 # related to solver iterations essentially?,
            ),
            viewer_options = gs.options.ViewerOptions(
                camera_pos    = (0.5, -0.5, 0.5),
                camera_lookat = (0.0, 0.0, 0.25),
                camera_fov    = 40,
                max_FPS = 60,
            ),
            rigid_options = gs.options.RigidOptions(
                enable_self_collision = enable_self_collision,
            ),
        )

        self.cam = self.scene.add_camera(
            res    = (512, 512),
            pos    = (2.5, -0.5, 1.0),
            lookat = (0.0, 0.0, 0.25),
            fov    = np.rad2deg(0.63),
            GUI    = False
        )
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size = (0.04, 0.04, 0.04),
                pos = (0.6, 0.0, 0.02),
            ),
        )
        self.scene.build(n_envs=self.num_envs, env_spacing=(5.0, 5.0))

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
            self.robot.set_dofs_kp(
                np.array([1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3]),
            )
            self.robot.set_dofs_kv(
                np.array([1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2]),
            )

        if self.control_mode == "pd_joint_delta_pos":
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_envs, 9))
            self.single_action_space = gym.spaces.Box(low=-1, high=1, shape=(9, ))


        self.fixed_trajectory = {
            "pick_and_lift": {
                "control_mode": "pd_joint_pos",
                "actions": [(torch.tensor([0.0, 0.68, 0.0, -1.9292649, 0.0, 2.627549, 0.7840855, 0.04, 0.04], device=gs.device), 5),
                            (torch.tensor([0.0, 0.68, 0.0, -1.9292649, 0.0, 2.627549, 0.7840855, -0.01, -0.01], device=gs.device), 5),
                            (torch.tensor([0.0, 0.35, 0.0, -1.9292649, 0.0, 2.627549, 0.7840855, -0.01, -0.01], device=gs.device), 5),
                            ],
                "shake_steps": 85,
                "obs": None,
            }
        }

    def set_control_mode(self, control_mode: str):
        self.control_mode = control_mode

    def get_obs(self):
        qpos = self.robot.get_dofs_position(self.motor_dofs)
        qvel = self.robot.get_dofs_velocity(self.motor_dofs)
        obs_buf = torch.cat(
            [
                qpos,
                qvel,
            ],
            axis=-1,
        )
        return obs_buf
    def step(self, action):
        if self.control_mode == "pd_joint_delta_pos":
            action = torch.clip(action, -1, 1)
            # match maniskill action scaling
            arm_action = action[:, :7] * 0.1
            gripper_action = action[:, 7:] * 0.05 - 0.01
            self.robot.control_dofs_position(arm_action + self.robot.get_dofs_position(self.arm_dofs), self.arm_dofs)
            self.robot.control_dofs_position(gripper_action + self.robot.get_dofs_position(self.gripper_dofs), self.gripper_dofs)
        elif self.control_mode == "pd_joint_pos":
            target_qpos = action
            self.robot.control_dofs_position(target_qpos)
        for _ in range(self.sim_steps_per_control_step):
            self.scene.step()
        return self.get_obs(), None, None, None, {}
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self.robot.set_dofs_position(torch.tile(self.rest_qpos, (self.num_envs, 1)), zero_velocity=True,)
        return self.get_obs(), {}
    def render_rgb_array(self):
        rgb = self.cam.render(rgb=True)[0]
        return rgb
