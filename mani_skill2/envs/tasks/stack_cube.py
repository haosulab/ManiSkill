from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import torch

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.envs.utils.randomization.pose import random_quaternions
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.building.actors import build_cube
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.utils.scene_builder.table.table_scene_builder import TableSceneBuilder
from mani_skill2.utils.structs.pose import Pose


@register_env(name="StackCube-v1", max_episode_steps=100)
class StackCubeEnv(BaseEnv):
    def __init__(self, *args, robot_uid="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uid=robot_uid, **kwargs)

    def _register_sensors(self):
        pose = look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10)
        ]

    def _register_render_cameras(self):
        pose = look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10)

    def _load_actors(self):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.box_half_size = torch.Tensor([0.02] * 3, device=self.device)
        self.cubeA = build_cube(
            self._scene, half_size=0.02, color=[1, 0, 0, 1], name="cubeA"
        )
        self.cubeB = build_cube(
            self._scene, half_size=0.02, color=[0, 1, 0, 1], name="cubeB"
        )

    def _initialize_actors(self):
        self.table_scene.initialize()
        qs = random_quaternions(
            self._episode_rng, lock_x=True, lock_y=True, lock_z=False, n=self.num_envs
        )
        ps = [0, 0, 0.02]
        self.cubeA.set_pose(Pose.create_from_pq(p=ps, q=qs))

    def _get_obs_extra(self):
        return OrderedDict()

    def evaluate(self, obs: Any):
        return {"success": torch.zeros(self.num_envs, device=self.device, dtype=bool)}

    def compute_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
