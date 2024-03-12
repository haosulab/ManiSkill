from collections import OrderedDict
from typing import Dict, Union

import numpy as np
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.scene_builder.table.table_scene_builder import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array


@register_env("PullCube-v1", max_episode_steps=50)
class PullCubeEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["sparse", "none"]

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Xmate3Robotiq, Fetch]
    goal_radius = 0.1
    cube_half_size = 0.02

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _sensor_configs(self):
        pose = look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _human_render_camera_configs(self):
        pose = look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # create cube
        self.obj = actors.build_cube(
            self._scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
        )

        # create target
        self.goal_region = actors.build_red_white_target(
            self._scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
        )

    def _initialize_episode(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.cube_half_size
            q = [1, 0, 0, 0]

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

            target_region_xyz = xyz - torch.tensor([0.1 + self.goal_radius, 0, 0])

            target_region_xyz[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(
                self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )

        return {
            "success": is_obj_placed,
        }

    def _get_obs_extra(self, info: Dict):
        obs = OrderedDict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_region.pose.p,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
            )
        return obs

    # TODO (fix the reward for pull cube)
    # def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
    #     tcp_push_pose = Pose.create_from_pq(
    #         p=self.obj.pose.p
    #         + torch.tensor([-self.cube_half_size - 0.005, 0, 0], device=self.device)
    #     )
    #     tcp_to_push_pose = tcp_push_pose.p - self.agent.tcp.pose.p
    #     tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
    #     reaching_reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)
    #     reward = reaching_reward

    #     reached = tcp_to_push_pose_dist < 0.01
    #     obj_to_goal_dist = torch.linalg.norm(
    #         self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
    #     )
    #     place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
    #     reward += place_reward * reached

    #     reward[info["success"]] = 3
    #     return reward

    # def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
    #     max_reward = 3.0
    #     return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
