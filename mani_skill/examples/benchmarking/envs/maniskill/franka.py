import os
from typing import Dict

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig
from transforms3d.euler import euler2quat

@register_env("FrankaBenchmark-v1", max_episode_steps=200000)
class FrankaBenchmarkEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none"]
    """
    This is just a dummy environment for showcasing robots in a empty scene
    """

    def __init__(self, *args, camera_width=128, camera_height=128, num_cameras=1, **kwargs):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.num_cameras = num_cameras
        super().__init__(*args, robot_uids="panda", **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=120,
            spacing=5,
            control_freq=60,
            scene_config=SceneConfig(
                bounce_threshold=0.5,
                solver_position_iterations=8, solver_velocity_iterations=0
            ),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien.Pose((-0.4, 0, 1.0), euler2quat(0, np.deg2rad(28.648), 0))
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
        return dict()

    def _load_scene(self, options: dict):

        # texture_square_len should be 4, but IsaacLab has a franka robot model that seems? to be larger than normal.
        # instead of shrinking robot size we shrink the ground texture to visually match the isaac lab franka robot setup.
        self.ground = build_ground(self.scene, texture_file=os.path.join(os.path.dirname(__file__), "assets/black_grid.png"), texture_square_len=2, mipmap_levels=6)

        # self.ground.set_collision_group_bit(group=2, bit_idx=30, bit=1)
        pass

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            qpos = self.agent.keyframes["rest"].qpos
            qpos[0] = 0.5
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose(p=[1.5, 0, 0], q=euler2quat(0, 0, np.pi)))
    def _load_lighting(self, options: Dict):
        self.scene.set_ambient_light(np.array([1,1,1])*0.1)
        for i in range(self.num_envs):
            self.scene.sub_scenes[i].set_environment_map(os.path.join(os.path.dirname(__file__), "kloofendal_28d_misty_puresky_1k.hdr"))
        self.scene.add_directional_light(
            [0.3, 0.3, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=2048
        )
    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: Dict):
        return dict()
