from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import torch

import mani_skill2.envs.utils.randomization as randomization
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.building.actors import build_cube, build_sphere
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.utils.scene_builder.table.table_scene_builder import TableSceneBuilder
from mani_skill2.utils.structs.pose import Pose


@register_env("PickCube-v1", max_episode_steps=100)
class PickCubeEnv(BaseEnv):
    """
    Task Description
    ----------------
    A simple task where the objective is to grasp a cube and move it to a target goal position.

    Randomizations
    --------------
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the cube's z-axis rotation is randomized to a random angle
    - the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]


    Success Conditions
    ------------------
    - the cube position is within goal_thresh (default 0.025) euclidean distance of the goal position

    Visualization: TODO: ADD LINK HERE

    Changelog:
    Different to v0, v1 does not require the robot to be static at the end which makes this task similar to other benchmarks and also easier
    """

    cube_half_size = 0.02
    goal_thresh = 0.025

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
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = build_cube(
            self._scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cube"
        )
        self.goal_site = build_sphere(
            self._scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
        )

    def _initialize_actors(self):
        self.table_scene.initialize()
        xyz = np.zeros((self.num_envs, 3))
        xyz[:, :2] = self._episode_rng.uniform(-0.1, 0.1, [self.num_envs, 2])
        xyz[:, 2] = self.cube_half_size
        qs = randomization.random_quaternions(
            self._episode_rng, lock_x=True, lock_y=True, n=self.num_envs
        )
        self.cube.set_pose(Pose.create_from_pq(xyz, qs, device=self.device))

        goal_xyz = np.zeros((self.num_envs, 3))
        goal_xyz[:, :2] = self._episode_rng.uniform(-0.1, 0.1, [self.num_envs, 2])
        goal_xyz[:, 2] = self._episode_rng.uniform(0, 0.3, [self.num_envs]) + xyz[:, 2]
        self.goal_site.set_pose(Pose.create_from_pq(goal_xyz, device=self.device))

    def _get_obs_extra(self):
        obs = OrderedDict(tcp_pose=self.agent.tcp.pose, goal_pos=self.goal_site)
        if "state" in self.obs_mode:
            obs.update(obs_pose=self.cube.pose.raw_pose)

    def evaluate(self, obs: Any):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        return {"success": is_obj_placed, "is_obj_placed": is_obj_placed}

    def compute_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = self.agent.is_grasping(self.cube)
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        reward[info["success"]] = 4
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4
