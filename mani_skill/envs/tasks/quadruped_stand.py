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


# @register_env("QuadrupedStand-v1", max_episode_steps=200)
class QuadrupedStandEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["anymal-c"]
    agent: ANYmalC

    def __init__(self, *args, robot_uids="anymal-c", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
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
                mount=self.agent.robot.links[0],
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
            mount=self.agent.robot.links[0],
        )

    def _load_scene(self):
        # for i in range(10):
        #     ground = build_ground(self._scene, return_builder=True)
        #     ground.initial_pose = sapien.Pose(p=[i * 40, 0, 0])
        #     ground.build_static(name="ground")
        # self.ground = self._scene.create_actor_builder()
        self.ground = build_meter_ground(self._scene, floor_width=1000)
        self.cube = actors.build_cube(
            self._scene, 0.05, color=[1, 0, 0, 1], name="cube"
        )
        # TODO (stao): why is this collision mesh so wacky?
        # mesh = self.agent.robot.get_collision_mesh(first_only=True)
        # self.height = -mesh[0].bounding_box.bounds[0, 2]
        self.height = 1.626

    def _initialize_episode(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            self.agent.robot.set_pose(Pose.create_from_pq(p=[0, 0, self.height]))
            self.agent.reset(init_qpos=torch.zeros(self.agent.robot.max_dof))
            self.cube.set_pose(Pose.create_from_pq(p=[0, 0, 0.05]))

    def evaluate(self):
        forces = self.agent.robot.get_net_contact_forces(["RH_KFE", "LH_KFE"]).norm(
            dim=(1, 2)
        )
        return {"success": self.agent.is_standing()}

    def _get_obs_extra(self, info: Dict):
        return OrderedDict(robot_pose=self.agent.robot.pose.raw_pose)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        reward = info["success"]
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
