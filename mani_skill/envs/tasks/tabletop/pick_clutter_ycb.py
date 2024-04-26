import os
from typing import Any, Dict, List, Union

import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.table_scene_builder import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


#
class PickClutterEnv(BaseEnv):
    """Base environment picking items out of clutter type of tasks. Flexibly supports using different configurations and object datasets"""

    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    DEFAULT_EPISODE_JSON: str
    DEFAULT_ASSET_ROOT: str
    DEFAULT_MODEL_JSON: str

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        episode_json: str = None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise

        if episode_json is None:
            episode_json = self.DEFAULT_EPISODE_JSON
        if not os.path.exists(episode_json):
            raise FileNotFoundError(
                f"Episode json ({episode_json}) is not found."
                "To download default json:"
                "`python -m mani_skill.utils.download_asset pick_clutter_ycb`."
            )
        self._episodes: List[Dict] = load_json(episode_json)

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

        if self.num_envs == 1:
            # with just one environment there isn't going to be a lot of geometrical variation
            # so setting the freq below to 1 ensures each reset (while a little slower) changes the loaded geometries
            self.reconfiguration_freq = 1

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_cfg=GPUMemoryConfig(
                max_rigid_contact_count=2**21, max_rigid_patch_count=2**19
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_model(self, model_id: str) -> ActorBuilder:
        raise NotImplementedError()

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        # sample some clutter configurations
        eps_idxs = np.arange(0, len(self._episodes))
        rand_idx = torch.randperm(len(eps_idxs), device=torch.device("cpu"))
        eps_idxs = eps_idxs[rand_idx]
        eps_idxs = np.concatenate(
            [eps_idxs] * np.ceil(self.num_envs / len(eps_idxs)).astype(int)
        )[: self.num_envs]

        self.selectable_target_objects: List[List[Actor]] = []
        """for each sub-scene, a list of objects that can be selected as targets"""
        all_objects = []

        for i, eps_idx in enumerate(eps_idxs):
            self.selectable_target_objects.append([])
            episode = self._episodes[eps_idx]
            for actor_cfg in episode["actors"]:
                builder = self._load_model(actor_cfg["model_id"])
                init_pose = actor_cfg["pose"]
                builder.initial_pose = sapien.Pose(p=init_pose[:3], q=init_pose[3:])
                builder.set_scene_idxs([i])
                obj = builder.build(name=f"set_{i}_{actor_cfg['model_id']}")
                all_objects.append(obj)
                if actor_cfg["rep_pts"] is not None:
                    # rep_pts is representative points, representing visible points
                    # we only permit selecting target objects that are visible
                    self.selectable_target_objects[-1].append(obj)

        self.all_objects = Actor.merge(all_objects, name="all_objects")

        self.goal_site = actors.build_sphere(
            self._scene,
            radius=0.01,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
        )
        self._hidden_objects.append(self.goal_site)

        self._sample_target_objects()

    def _sample_target_objects(self):
        # note this samples new target objects for every sub-scene
        target_objects = []
        for i in range(self.num_envs):
            selected_obj_idxs = torch.randint(low=0, high=99999, size=(self.num_envs,))
            selected_obj_idxs[i] = selected_obj_idxs[i] % len(
                self.selectable_target_objects[-1]
            )
            target_objects.append(
                self.selectable_target_objects[-1][selected_obj_idxs[i]]
            )
        self.target_object = Actor.merge(target_objects, name="target_object")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)
            goal_pos = torch.rand(size=(b, 3)) * torch.tensor(
                [0.3, 0.5, 0.1]
            ) + torch.tensor([-0.15, -0.25, 0.35])
            self.goal_pos = goal_pos
            self.goal_site.set_pose(Pose.create_from_pq(self.goal_pos))

            # reset objects to original poses
            if b == self.num_envs:
                # if all envs reset
                self.all_objects.pose = self.all_objects.initial_pose
            else:
                # if only some envs reset, we unfortunately still have to do some mask wrangling
                mask = torch.isin(self.all_objects._scene_idxs, env_idx)
                self.all_objects.pose = self.all_objects.initial_pose[mask]

    def evaluate(self):
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=bool),
            "fail": torch.zeros(self.num_envs, device=self.device, dtype=bool),
        }

    def _get_obs_extra(self, info: Dict):
        return dict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward


@register_env("PickClutterYCB-v1", max_episode_steps=100)
class PickClutterYCBEnv(PickClutterEnv):
    DEFAULT_EPISODE_JSON = f"{ASSET_DIR}/tasks/pick_clutter/ycb_train_5k.json.gz"

    def _load_model(self, model_id):
        builder, _ = actors.build_actor_ycb(
            model_id, self._scene, name=model_id, return_builder=True
        )
        return builder
