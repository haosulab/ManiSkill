from typing import Any, List, Union
import genesis as gs
from genesis.engine.entities import RigidEntity
from .base_env import BaseEnv
import gymnasium as gym
import numpy as np
import torch
class FrankaPickCubeBenchmarkEnv(BaseEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}
    def __init__(self, num_envs: int, sim_freq: int, control_freq: int, render_mode: str, control_mode: str = "pd_joint_delta_pos", robot_uids: Union[str, List[str]] = "panda"):
        super().__init__(
            num_envs,
            # NOTE (stao): it's unclear what the right solver parameters are. Documentation suggests substeps=4 but that doesn't work properly
            # experimented with different integrators (mjx pick cube env uses implicitfast) and different number of iterations but to no avail
            sim_options=gs.options.SimOptions(dt=1/sim_freq, substeps=4),
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
        self.fixed_trajectory = {
            "pick_and_lift": {
                "control_mode": "pd_joint_pos",
                "actions": [(torch.tensor([0.0, 0.68, 0.0, -1.9292649, 0.0, 2.627549, 0.7840855, 0.04, 0.04], device=gs.device), 15),
                            (torch.tensor([0.0, 0.68, 0.0, -1.9292649, 0.0, 2.627549, 0.7840855, -0.02, -0.02], device=gs.device), 15),
                            (torch.tensor([0.0, 0.3, 0.0, -1.9292649, 0.0, 2.627549, 0.7840855, -0.02, -0.02], device=gs.device), 20),
                            ],
                "shake_action_fn": lambda : torch.cat([torch.rand(self.num_envs, 7, device=gs.device) * 2 - 1, torch.ones(self.num_envs, 2, device=gs.device) * -1], dim=-1),
                "shake_steps": 150,
            },
        }
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

    def _load_scene(self):
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.robot: RigidEntity = self.scene.add_entity(
            gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size = (0.04, 0.04, 0.04),
                pos = (0.6, 0.0, 0.02),
            ),
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
        self.robot.set_qpos(torch.tile(self.rest_qpos, (self.num_envs, 1)), zero_velocity=True,)
        self.cube.set_pos(torch.tile(torch.tensor([0.6, 0.0, 0.02], device=gs.device), (self.num_envs, 1)))
        self.cube.set_quat(torch.tile(torch.tensor([1, 0, 0, 0], device=gs.device), (self.num_envs, 1)))
