from collections import OrderedDict
from typing import Any, Dict

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
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("Empty-v1", max_episode_steps=200000)
class EmptyEnv(BaseEnv):
    """
    This is just a dummy environment for showcasing robots in a empty scene
    """

    SUPPORTED_ROBOTS = ["panda", "fetch", "xmate3_robotiq", "anymal"]
    # agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.75, -0.75, 0.5], [0.0, 0.0, 0.2])
        return CameraConfig("render_camera", pose, 2048, 2048, 1, 0.01, 100)

    def _load_scene(self):
        build_ground(self._scene)

    def _initialize_episode(self, env_idx: torch.Tensor):
        if self.robot_uids == "panda":
            qpos = np.array(
                [
                    0.0,
                    0.0,
                    0,
                    -np.pi * 6 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            self.agent.robot.set_qpos(qpos)
        elif self.robot_uids == "xmate3_robotiq":
            qpos = np.array(
                [0, np.pi / 6, 0, np.pi / 3, 0, np.pi / 2, -np.pi / 2, 0, 0]
            )
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.562, 0, 0]))
        elif self.robot_uids == "fetch":
            qpos = np.array(
                [
                    0,
                    0,
                    0,
                    0.04,
                    0,
                    0,
                    0,
                    -np.pi / 4,
                    0,
                    np.pi / 4,
                    0,
                    np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.agent.robot.set_qpos(qpos)

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: Dict):
        return OrderedDict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
