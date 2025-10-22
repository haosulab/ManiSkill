import os
from typing import Dict

import numpy as np
from mani_skill.utils.building import actors
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig
from transforms3d.euler import euler2quat

@register_env("FrankaPickCubeBenchmark-v1")
class FrankaPickCubeBenchmarkEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none"]
    def __init__(self, *args, camera_width=128, camera_height=128, num_cameras=1, **kwargs):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.num_cameras = num_cameras
        super().__init__(*args, robot_uids="panda", **kwargs)

        self.fixed_trajectory = {
            "pick_and_lift": {
                "control_mode": "pd_joint_pos",
                "actions": [(torch.tensor([0.0, 0.68, 0.0, -1.9292649, 0.0, 2.627549, 0.7840855, 0.04], device=self.device), 15),
                            (torch.tensor([0.0, 0.68, 0.0, -1.9292649, 0.0, 2.627549, 0.7840855, -0.02], device=self.device), 15),
                            (torch.tensor([0.0, 0.3, 0.0, -1.9292649, 0.0, 2.627549, 0.7840855, -0.02], device=self.device), 20),
                            ],
                "shake_action_fn": lambda : torch.cat([torch.rand(self.num_envs, 7, device=self.device) * 0.5 - 0.25, torch.ones(self.num_envs, 1, device=self.device) * -1], dim=-1),
                "shake_steps": 150,
            },
        }

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=100,
            spacing=5,
            control_freq=50,
            scene_config=SceneConfig(
                solver_position_iterations=10, solver_velocity_iterations=1
            ),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at((2.5, -0.5, 1.0), (0, 0, 0.25))
        sensor_configs = []
        if self.num_cameras is not None:
            for i in range(self.num_cameras):
                sensor_configs.append(CameraConfig(uid=f"base_camera_{i}",
                                                pose=pose,
                                                width=self.camera_width,
                                                height=self.camera_height,
                                                far=25,
                                                fov=0.63))
        return sensor_configs
    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig(
            uid="render_camera",
            pose=sapien_utils.look_at((2.5, -0.5, 1.0), (0, 0, 0.25)),
            width=512,
            height=512,
            far=25,
            fov=0.63,
        )
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))
    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene, texture_square_len=2, mipmap_levels=6)
        self.cube = actors.build_box(self.scene, half_sizes=(0.02, 0.02, 0.02), color=[1, 0, 0, 1], name="cube", initial_pose=sapien.Pose(p=[0.6, 0, 0.02]))
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            qpos = self.agent.keyframes["rest"].qpos
            self.agent.robot.set_qpos(qpos)
            self.cube.set_pose(sapien.Pose(p=[0.6, 0, 0.02]))
    def _load_lighting(self, options: dict):
        # self.scene.set_ambient_light(np.array([1,1,1])*0.05)
        for i in range(self.num_envs):
            self.scene.sub_scenes[i].set_environment_map(os.path.join(os.path.dirname(__file__), "kloofendal_28d_misty_puresky_1k.hdr"))
        self.scene.add_directional_light(
            [0.3, 0.3, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=2048
        )
    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: dict):
        return dict()
