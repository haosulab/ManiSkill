from typing import Dict

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.fetch.fetch import FETCH_WHEELS_COLLISION_BIT, Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("RoboCasaKitchen-v1", max_episode_steps=100)
class RoboCasaKitchenEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none"]
    """
    This is a scene sampled from the RoboCasa dataset that you can explore and take random actions in. No rewards/success metrics are defined.
    """

    def __init__(self, *args, robot_uids="fetch", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(spacing=8)

    @property
    def _default_sensor_configs(self):
        # TODO (fix cameras to be where robocasa places them)
        pose = sapien_utils.look_at([1.25, -1.25, 1.5], [0.0, 0.0, 0.2])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.25, -1.25, 1.5], [0.0, 0.0, 0.2])
        return CameraConfig("render_camera", pose, 2048, 2048, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.scene_builder = RoboCasaSceneBuilder(self)
        data = self.scene_builder.build()
        self.fixtures = data["fixtures"]
        # self.actors = data["actors"]
        self.fixture_configs = data["fixture_configs"]
        # self.ground = build_ground(self.scene)
        # self.ground.set_collision_group_bit(
        #     group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
        # )
        # TODO (stao): disable wheel collisions with floors.

        self.agent.robot.initial_pose = sapien.Pose(p=[2.7, -1.5, 0])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
            self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)
            self.agent.robot.set_pose(sapien.Pose(p=[2.7, -1.5, 0]))

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: Dict):
        return dict()