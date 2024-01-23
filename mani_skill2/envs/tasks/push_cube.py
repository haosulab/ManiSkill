"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill2 tasks can minimally be defined by how the environment resets, what agents/objects are
loaded, goal parameterization, and success conditions

Environment reset is comprised of running two functions, `self.reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill2. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the positions of all objects (called actors), articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do
"""

from collections import OrderedDict
from typing import Any, Dict, Union

import numpy as np
import torch
from transforms3d.euler import euler2quat

from mani_skill2.agents.robots.panda.panda import Panda
from mani_skill2.agents.robots.xmate3.xmate3 import Xmate3Robotiq
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.building.actors import build_cube, build_red_white_target
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import (  # import various useful utilities for working with sapien
    look_at,
)
from mani_skill2.utils.scene_builder.table.table_scene_builder import TableSceneBuilder
from mani_skill2.utils.structs.pose import Pose, vectorize_pose
from mani_skill2.utils.structs.types import Array


@register_env("PushCube-v0", max_episode_steps=50)
class PushCubeEnv(BaseEnv):
    # Specify some supported robot types
    agent: Union[Panda, Xmate3Robotiq]

    # set some commonly used values
    goal_radius = 0.1
    cube_half_size = 0.02

    def __init__(self, *args, robot_uid="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uid=robot_uid, **kwargs)

    def _register_sensors(self):
        # registers one camera looking at the robot, cube, and target
        pose = look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10)
        ]

    def _register_render_cameras(self):
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        pose = look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10)

    def _load_actors(self):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # we then add the cube that we want to push and give it a color and size using a convenience build_cube function
        # we specify the body_type to be "dynamic" as it should be able to move when touched by other objects / the robot
        self.obj = build_cube(
            self._scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
        )

        # we also add in red/white target to visualize where we want the cube to be pushed to
        # we specify add_collisions=False as we only use this as a visual for videos and do not want it to affect the actual physics
        # we finally specify the body_type to be "kinematic" so that the object stays in place
        self.goal_region = build_red_white_target(
            self._scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
        )

    def _initialize_actors(self):
        # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
        # note that the table scene is built such that z=0 is the surface of the table.
        self.table_scene.initialize()

        # here we write some randomization code that randomizes the x, y position of the cube we are pushing in the range [-0.1, -0.1] to [0.1, 0.1]
        xyz = torch.zeros((self.num_envs, 3), device=self.device)
        xyz[..., :2] = torch.from_numpy(
            self._episode_rng.uniform(-0.1, 0.1, [self.num_envs, 2])
        ).cuda()
        xyz[..., 2] = self.cube_half_size
        q = [1, 0, 0, 0]
        # we can then create a pose object using Pose.create_from_pq to then set the cube pose with. Note that even though our quaternion
        # is not batched, Pose.create_from_pq will automatically batch p or q accordingly
        obj_pose = Pose.create_from_pq(p=xyz, q=q)
        self.obj.set_pose(obj_pose)

        # here we set the location of that red/white target (the goal region). In particular here, we set the position to be in front of the cube
        # and we further rotate 90 degrees on the y-axis to make the target object face up
        target_region_xyz = xyz + torch.tensor(
            [0.1 + self.goal_radius, 0, 0], device=self.device
        )
        target_region_xyz[
            ..., 2
        ] = 1e-3  # set a little bit above 0 so the target is sitting on the table
        self.goal_region.set_pose(
            Pose.create_from_pq(
                p=target_region_xyz,
                q=euler2quat(0, np.pi / 2, 0),
            )
        )

    def _get_obs_extra(self):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.agent.tcp.pose),
            goal_pos=self.goal_region.pose.p,
        )
        if self._obs_mode in ["state", "state_dict"]:
            # if the observation mode is state/state_dict, we provide ground truth information about where the cube is.
            # for visual observation modes one should rely on the sensed visual data to determine where the cube is
            obs.update(
                obj_pose=vectorize_pose(self.obj.pose),
            )
        return obs

    def evaluate(self, obs: Any):
        # success is achieved when the cube's xy position on the table is within the
        # goal region's area (a circle centered at the goal region's xy position)
        is_obj_placed = (
            torch.linalg.norm(
                self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )

        return {
            "success": is_obj_placed,
        }

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # We also create a pose marking where the robot should push the cube from that is easiest (pushing from behind the cube)
        tcp_push_pose = Pose.create_from_pq(
            p=self.obj.pose.p
            + torch.tensor([-self.cube_half_size - 0.005, 0, 0], device=self.device)
        )
        tcp_to_push_pose = tcp_push_pose.p - self.agent.tcp.pose.p
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)
        reward = reaching_reward

        # compute a placement reward to encourage robot to move the cube to the center of the goal region
        # we further multiply the place_reward by a mask reached so we only add the place reward if the robot has reached the desired push pose
        # This reward design helps train RL agents faster by staging the reward out.
        reached = tcp_to_push_pose_dist < 0.01
        obj_to_goal_dist = torch.linalg.norm(
            self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * reached

        # assign rewards to parallel environments that achieved success to the maximum of 3.
        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
