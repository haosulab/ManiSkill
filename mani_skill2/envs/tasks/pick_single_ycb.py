from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np
import torch

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.envs.utils.randomization.pose import random_quaternions
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.building.actors import (
    MODEL_DBS,
    _load_ycb_dataset,
    build_actor_ycb,
)
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.utils.scene_builder.table.table_scene_builder import TableSceneBuilder
from mani_skill2.utils.structs.actor import Actor
from mani_skill2.utils.structs.pose import Pose


@register_env("PickSingleYCB-v1", max_episode_steps=100)
class PickSingleYCBEnv(BaseEnv):
    """
    Task Description
    ----------------
    Pick up a random object sampled from the [YCB dataset](https://www.ycbbenchmarks.com/) and move it to a random goal position

    Randomizations
    --------------

    Success Conditions
    ------------------

    Visualization: link to a video/gif of the task being solved
    """

    def __init__(self, *args, robot_uid="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.model_id = None
        _load_ycb_dataset()
        self.all_model_ids = np.array(list(MODEL_DBS["YCB"]["model_data"].keys()))
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

        # randomize the list of all possible models in the YCB dataset
        # then sub-scene i will load model model_ids[i % number_of_ycb_objects]
        rand_idx = torch.randperm(len(self.all_model_ids))
        model_ids = self.all_model_ids[rand_idx]
        model_ids = np.concatenate(
            [model_ids] * np.ceil(self.num_envs / len(self.all_model_ids)).astype(int)
        )[: self.num_envs]
        if self.num_envs < len(self.all_model_ids):
            print(
                "There are less parallel environments than total available models to sample. The environment will run considerably slower"
            )

        actors: List[Actor] = []
        self.obj_heights = []
        for i, model_id in enumerate(model_ids):
            builder, obj_height = build_actor_ycb(
                model_id, self._scene, name=model_id, return_builder=True
            )
            scene_mask = np.zeros(self.num_envs, dtype=bool)
            scene_mask[i] = True
            builder.set_scene_mask(scene_mask)
            actors.append(builder.build(name=f"{model_id}-{i}"))
            self.obj_heights.append(obj_height)
        self.obj = Actor.merge_actors(actors, name="ycb_object")

    def _initialize_actors(self):
        with torch.device(self.device):
            self.table_scene.initialize()
            ps = torch.zeros((self.num_envs, 3))
            for i in range(self.num_envs):
                # use ycb object bounding box heights to set it properly on the table
                ps[i, 2] = self.obj_heights[i] / 2

            qs = random_quaternions(self.num_envs, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=ps, q=qs))

    def _get_obs_extra(self):
        return OrderedDict()

    def evaluate(self, obs: Any):
        return {"success": torch.zeros(self.num_envs, device=self.device, dtype=bool)}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
