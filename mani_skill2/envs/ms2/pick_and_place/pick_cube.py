from collections import OrderedDict
from typing import Dict

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import to_tensor
from mani_skill2.utils.scene_builder import TableSceneBuilder
from mani_skill2.utils.structs.pose import Pose, vectorize_pose

from .base_env import StationaryManipulationEnv


@register_env("PickCube-v0", max_episode_steps=100)
class PickCubeEnv(StationaryManipulationEnv):
    goal_thresh = 0.025
    min_goal_dist = 0.05

    def __init__(self, *args, obj_init_rot_z=True, **kwargs):
        self.obj_init_rot_z = obj_init_rot_z
        self.cube_half_size = np.array([0.02] * 3, np.float32)
        super().__init__(*args, **kwargs)

    def _load_actors(self):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.obj = self._build_cube(self.cube_half_size)
        self.goal_site = self._build_sphere_site(self.goal_thresh)

    def _initialize_actors(self):
        self.table_scene.initialize()
        xyz = np.zeros((self.num_envs, 3))
        xyz[..., :2] = self._episode_rng.uniform(-0.1, 0.1, [self.num_envs, 2])
        xyz[..., 2] = self.cube_half_size[2]
        qs = [1, 0, 0, 0]
        if self.obj_init_rot_z:
            qs = []
            for i in range(self.num_envs):
                ori = self._episode_rng.uniform(0, 2 * np.pi)
                q = euler2quat(0, 0, ori)
                qs.append(q)
            qs = to_tensor(qs)
        # to set a batch of poses, use the Pose object or provide a raw tensor
        obj_pose = Pose.create_from_pq(p=xyz, q=qs)

        self.obj.set_pose(obj_pose)

    def _initialize_task(self, max_trials=100, verbose=False):
        obj_pos = self.obj.pose.p
        # Sample a goal position far enough from the object

        # TODO (stao): Make this code batched.
        goal_poss = []
        for j in range(len(obj_pos)):
            for i in range(max_trials):
                goal_xy = self._episode_rng.uniform(-0.1, 0.1, [2])
                goal_z = self._episode_rng.uniform(0, 0.5) + obj_pos[j, 2]
                goal_pos = torch.hstack([to_tensor(goal_xy), goal_z])
                if torch.linalg.norm(goal_pos - obj_pos) > self.min_goal_dist:
                    if verbose:
                        print(f"Found a valid goal at {i}-th trial")
                    goal_poss.append(goal_pos)
                    break
        self.goal_pos = torch.vstack(goal_poss)
        self.goal_site.set_pose(Pose.create_from_pq(self.goal_pos))

    def _get_obs_extra(self, info: Dict) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.agent.tcp.pose),
            goal_pos=self.goal_pos,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                tcp_to_goal_pos=self.goal_pos - self.agent.tcp.pose.p,
                obj_pose=vectorize_pose(self.obj.pose),
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_pos - self.obj.pose.p,
            )
        return obs

    def check_obj_placed(self):
        return (
            torch.linalg.norm(self.goal_pos - self.obj.pose.p, axis=1)
            <= self.goal_thresh
        )

    def check_robot_static(self, thresh=0.2):
        # Assume that the last two DoF is gripper
        qvel = self.agent.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= thresh

    def evaluate(self, **kwargs):
        is_obj_placed = self.check_obj_placed()
        is_robot_static = self.check_robot_static()
        tcp_to_obj_pos = self.obj.pose.p - self.agent.tcp.pose.p
        tcp_to_obj_dist = torch.linalg.norm(tcp_to_obj_pos, axis=1)
        is_grasped = self.agent.is_grasping(self.obj)
        obj_to_goal_dist = torch.linalg.norm(self.goal_pos - self.obj.pose.p, axis=1)
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        return dict(
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            tcp_to_obj_dist=tcp_to_obj_dist,
            is_grasped=is_grasped,
            place_reward=is_grasped * place_reward,
            success=torch.logical_and(is_obj_placed, is_robot_static),
        )

    def compute_dense_reward(self, obs, action, info):
        tcp_to_obj_dist = info["tcp_to_obj_dist"]
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(self.goal_pos - self.obj.pose.p, axis=1)
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped  # add place reward only if we are grasping

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        reward += static_reward * info["is_obj_placed"] * info["is_grasped"]

        reward[info["success"]] = 5

        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0

    def render_human(self):
        self.goal_site.show_visual()
        ret = super().render_human()
        self.goal_site.hide_visual()
        return ret

    def render_rgb_array(self, *args, **kwargs):
        # self.goal_site.show_visual()
        ret = super().render_rgb_array(*args, **kwargs)
        # self.goal_site.hide_visual()
        return ret

    def render_cameras(self):
        self.goal_site.hide_visual()
        ret = super().render_cameras()
        self.goal_site.show_visual()
        return ret

    def _get_obs_images(self):
        self.goal_site.hide_visual()
        ret = super()._get_obs_images()
        self.goal_site.show_visual()
        return ret

    def get_state(self):
        state = super().get_state()
        return torch.hstack([state, self.goal_pos])

    def set_state(self, state):
        self.goal_pos = state[:, -3:]
        super().set_state(state[:, :-3])


@register_env("LiftCube-v0", max_episode_steps=200)
class LiftCubeEnv(PickCubeEnv):
    """Lift the cube to a certain height."""

    goal_height = 0.2

    def _initialize_task(self):
        self.goal_pos = self.obj.pose.p + torch.tensor(
            [0, 0, self.goal_height], device=self.device
        )
        self.goal_site.set_pose(Pose.create_from_pq(self.goal_pos))

    def _get_obs_extra(self, info: Dict) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.agent.tcp.pose),
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=vectorize_pose(self.obj.pose),
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def check_obj_placed(self):
        return self.obj.pose.p[..., 2] >= self.goal_height + self.cube_half_size[2]

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        # reaching reward
        gripper_pos = self.agent.tcp.pose.p
        obj_pos = self.obj.pose.p
        dist = torch.linalg.norm(gripper_pos - obj_pos, axis=1)
        reaching_reward = 1 - torch.tanh(5 * dist)
        reward += reaching_reward

        # is_grasped = self.agent.is_grasping(self.obj, max_angle=30)

        # # grasp reward
        # if is_grasped:
        #     reward += 0.25

        # lifting reward
        # if is_grasped:
        lifting_reward = self.obj.pose.p[..., 2] - self.cube_half_size[2]
        lifting_reward = torch.min(lifting_reward / self.goal_height, torch.tensor(1.0))
        reward += lifting_reward

        reward[info["success"]] = 2.25

        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 2.25
