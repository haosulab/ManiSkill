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
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


class QuadrupedStandEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    SUPPORTED_ROBOTS = ["anymal-c"]
    agent: ANYmalC

    def __init__(self, *args, robot_uids="anymal-c", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

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
                mount=self.agent.robot.links[0],
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([2.5, 2.5, 1], [0.0, 0.0, 0])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=512,
                height=512,
                fov=1,
                near=0.01,
                far=100,
                mount=self.agent.robot.links[0],
            )
        ]

    def _load_scene(self, options: dict):
        self.ground = build_ground(self._scene)
        self.cube = actors.build_cube(
            self._scene, 0.05, color=[1, 0, 0, 1], name="cube"
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.agent.robot.set_pose(Pose.create_from_pq(p=[0, 0, 1.2]))
            self.agent.reset(init_qpos=torch.zeros(self.agent.robot.max_dof))
            self.cube.set_pose(Pose.create_from_pq(p=[0, 0, 0.05]))

    def evaluate(self):
        is_standing = self.agent.is_standing()
        return {"fail": ~is_standing, "is_standing": is_standing}

    def _get_obs_extra(self, info: Dict):
        return dict(robot_pose=self.agent.robot.pose.raw_pose)

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return info["is_standing"]

    # def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
    #     reward = info["success"]
    #     return reward

    # def compute_normalized_dense_reward(
    #     self, obs: Any, action: torch.Tensor, info: Dict
    # ):
    #     max_reward = 1.0
    #     return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward


# @register_env("AnymalCStand-v1", max_episode_steps=200)
class AnymalCStandEnv(QuadrupedStandEnv):
    def __init__(self, *args, robot_uids="anymal-c", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
