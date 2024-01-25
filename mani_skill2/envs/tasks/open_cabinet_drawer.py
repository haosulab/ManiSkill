from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import torch

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.building.articulations import (
    MODEL_DBS,
    _load_partnet_mobility_dataset,
    build_preprocessed_partnet_mobility_articulation,
)
from mani_skill2.utils.building.ground import build_tesselated_square_floor
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.utils.structs.articulation import Articulation
from mani_skill2.utils.structs.pose import Pose


@register_env("OpenCabinet-v1", max_episode_steps=200)
class OpenCabinetEnv(BaseEnv):
    """
    Task Description
    ----------------
    Add a task description here

    Randomizations
    --------------

    Success Conditions
    ------------------

    Visualization: link to a video/gif of the task being solved
    """

    def __init__(
        self,
        *args,
        robot_uid="mobile_panda_single_arm",
        robot_init_qpos_noise=0.02,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        _load_partnet_mobility_dataset()
        self.all_model_ids = np.array(
            list(MODEL_DBS["PartnetMobility"]["model_data"].keys())
        )
        super().__init__(*args, robot_uid=robot_uid, **kwargs)

    def _register_sensors(self):
        pose = look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10)
        ]

    def _register_render_cameras(self):
        pose = look_at(eye=[-1.5, -1.5, 1.5], target=[-0.1, 0, 0.1])
        return CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10)

    def _load_actors(self):
        rand_idx = torch.randperm(len(self.all_model_ids))
        model_ids = self.all_model_ids[rand_idx]
        model_ids = np.concatenate(
            [model_ids] * np.ceil(self.num_envs / len(self.all_model_ids)).astype(int)
        )[: self.num_envs]
        self.ground = build_tesselated_square_floor(self._scene)

        cabinets = []
        self.cabinet_heights = []
        for i, model_id in enumerate(model_ids):
            scene_mask = np.zeros(self.num_envs, dtype=bool)
            scene_mask[i] = True
            cabinet, metadata = build_preprocessed_partnet_mobility_articulation(
                self._scene, model_id, name=f"{model_id}-{i}", scene_mask=scene_mask
            )
            self.cabinet_heights.append(
                metadata.bbox.bounds[1, 2] - metadata.bbox.bounds[0, 2]
            )
            # self.cabinet = cabinet
            # self.cabinet_metadata = metadata
            cabinets.append(cabinet)
        self.cabinet = Articulation.merge_articulations(cabinets, name="cabinet")
        self.cabinet_metadata = metadata

    def _initialize_actors(self):
        with torch.device(self.device):
            xyz = torch.zeros((self.num_envs, 3))
            xyz[:, 2] = torch.tensor(self.cabinet_heights) / 2
            self.cabinet.set_pose(Pose.create_from_pq(p=xyz))
            qlimits = self.cabinet.get_qlimits()  # [N, self.cabinet.max_dof, 2]
            qpos = qlimits[:, :, 0]
            self.cabinet.set_qpos(
                qpos
            )  # close all the cabinets. We know beforehand that lower qlimit means "closed" for these assets.
            # initialize robot
            if self.robot_uid == "panda":
                self.agent.robot.set_qpos(self.agent.robot.qpos * 0)
                self.agent.robot.set_pose(Pose.create_from_pq(p=[-1, 0, 0]))
            elif self.robot_uid == "mobile_panda_single_arm":
                center = np.array([0, 0.8])
                dist = self._episode_rng.uniform(1.6, 1.8)
                theta = self._episode_rng.uniform(0.9 * np.pi, 1.1 * np.pi)
                direction = np.array([np.cos(theta), np.sin(theta)])
                xy = center + direction * dist

                # Base orientation
                noise_ori = self._episode_rng.uniform(-0.05 * np.pi, 0.05 * np.pi)
                ori = (theta - np.pi) + noise_ori

                h = 1e-4
                arm_qpos = np.array([0, 0, 0, -1.5, 0, 3, 0.78, 0.02, 0.02])

                qpos = np.hstack([xy, ori, h, arm_qpos])
                self.agent.reset(qpos)

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
