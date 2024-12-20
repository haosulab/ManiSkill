from typing import Any, List, Union
import genesis as gs
from genesis.engine.entities import RigidEntity
from .base_env import BaseEnv
import gymnasium as gym
import numpy as np
import torch
class FrankaMoveBenchmarkEnv(BaseEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}
    def __init__(self, num_envs: int, sim_freq: int, control_freq: int, render_mode: str, control_mode: str = "pd_joint_delta_pos", robot_uids: Union[str, List[str]] = "panda"):
        super().__init__(
            num_envs,
            sim_options=gs.options.SimOptions(dt=1/sim_freq, substeps=1), # using less substeps here is faster but less accurate
            rigid_options=gs.options.RigidOptions(enable_self_collision=True),
            viewer_options=gs.options.ViewerOptions(
                camera_pos    = (0.5, -0.5, 0.5),
                camera_lookat = (0.0, 0.0, 0.25),
                camera_fov    = 40,
                max_FPS = 60,
            ),
            control_freq=control_freq,
            render_mode=render_mode,
            control_mode=control_mode,
            robot_uids=robot_uids,
        )
        self.fixed_trajectory = {}
        self.rest_qpos = torch.tensor(
            [
                0.5,
                np.pi / 8,
                0,
                -np.pi * 5 / 8,
                0,
                np.pi * 3 / 4,
                np.pi / 4,
                0.04,
                0.04,
            ], device=gs.device)

    def _load_scene(self):
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.robot: RigidEntity = self.scene.add_entity(
            gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
        )
    def _load_sensors(self):
        self.cam = self.scene.add_camera(
            res    = (512, 512),
            pos    = (2.5, -0.5, 1.0),
            lookat = (0.0, 0.0, 0.25),
            fov    = np.rad2deg(0.63),
            GUI    = False
        )
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
    def _initialize_episode(self):
        self.robot.set_dofs_position(torch.tile(self.rest_qpos, (self.num_envs, 1)), zero_velocity=True,)
