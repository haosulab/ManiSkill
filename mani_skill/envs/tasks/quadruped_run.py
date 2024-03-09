from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.anymal.anymal_c import ANYmalC
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground, build_meter_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig


# @register_env("QuadrupedRun-v1", max_episode_steps=200)
class QuadrupedRunEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["anymal-c"]
    agent: ANYmalC

    def __init__(self, *args, robot_uids="anymal-c", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    """
    NOTE that Isaac Anymal has these settings
    why is their patch/contact count so low compared to ours?
    gpu_max_rigid_contact_count: 524288
    gpu_max_rigid_patch_count: 163840
    gpu_found_lost_pairs_capacity: 4194304
    gpu_found_lost_aggregate_pairs_capacity: 33554432
    gpu_total_aggregate_pairs_capacity: 4194304
    """

    @property
    def _default_sim_cfg(self):
        return SimConfig(
            sim_freq=100,
            control_freq=50,
            gpu_memory_cfg=GPUMemoryConfig(
                heap_capacity=2**27,
                temp_buffer_capacity=2**25,
                found_lost_pairs_capacity=2**22,
                found_lost_aggregate_pairs_capacity=2**25,
                total_aggregate_pairs_capacity=2**22,
                max_rigid_patch_count=2**18,
                max_rigid_contact_count=2**20,
            ),
            scene_cfg=SceneConfig(
                solver_iterations=4,
                bounce_threshold=0.2,
            ),
        )

    @property
    def _sensor_configs(self):
        pose = sapien_utils.look_at([1.5, 1.5, 1], [0.0, 0.0, 0])
        return [
            CameraConfig(
                "base_camera",
                pose.p,
                pose.q,
                128,
                128,
                np.pi / 2,
                0.01,
                100,
                # link=self.agent.robot.links[0],
            )
        ]

    @property
    def _human_render_camera_configs(self):
        pose = sapien_utils.look_at([2.5, 2.5, 1], [0.0, 0.0, 0])
        return CameraConfig(
            "render_camera",
            pose.p,
            pose.q,
            512,
            512,
            1,
            0.01,
            100,
            # link=self.agent.robot.links[0],
        )

    def _load_scene(self):
        self.ground = build_meter_ground(self._scene, floor_width=20)
        self.height = 0.63

    def _initialize_episode(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            self.agent.robot.set_pose(Pose.create_from_pq(p=[0, 0, self.height]))
            self.agent.reset(init_qpos=torch.zeros(self.agent.robot.max_dof))

    def evaluate(self):
        return {"success": torch.zeros(self.num_envs, dtype=bool, device=self.device)}

    def _get_obs_extra(self, info: Dict):
        return OrderedDict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        xvel = self.agent.robot.root_linear_velocity[:, 0]
        # cts = self._scene.get_contacts()
        # print(len(cts))
        # for ct in cts:
        #     # if ct.bodies[1].entity.name == "base" or ct.bodies[0].entity.name == "base":
        #     print(ct)
        return xvel / 10

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
