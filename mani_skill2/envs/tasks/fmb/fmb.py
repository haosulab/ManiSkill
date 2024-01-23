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

import os.path as osp
from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill2.agents.robots import (
    ROBOTS,  # a dictionary mapping robot name to robot class that inherits BaseAgent
)
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.building.actor_builder import ActorBuilder
from mani_skill2.utils.building.actors import build_cube
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import (  # import various useful utilities for working with sapien
    look_at,
)
from mani_skill2.utils.scene_builder.table.table_scene_builder import TableSceneBuilder
from mani_skill2.utils.structs.pose import vectorize_pose


@register_env("FMBEnv-v0", max_episode_steps=200)
class FMBEnv(BaseEnv):
    # in the __init__ function you can pick a default robot your task should use e.g. the panda robot
    def __init__(self, *args, robot_uid="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uid=robot_uid, **kwargs)

    """
    Reconfiguration Code

    below are all functions involved in reconfiguration during environment reset called in the same order. As a user
    you can change these however you want for your desired task. These functions will only ever be called once in general. In CPU simulation,
    for some tasks these may need to be called multiple times if you need to swap out object assets. In GPU simulation these will only ever be called once.
    """

    def _register_sensors(self):
        # To customize the sensors that capture images/pointclouds for the environment observations,
        # simply define a CameraConfig as done below for Camera sensors. You can add multiple sensors by returning a list
        pose = look_at(
            eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1]
        )  # look_at is a utility to get the pose of a camera that looks at a target

        # to see what all the sensors capture in the environment for observations, run env.render_cameras() which returns an rgb array you can visualize
        return [
            CameraConfig("base_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10)
        ]

    def _register_render_cameras(self):
        pose = look_at([1.0, 0.8, 0.8], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose.p, pose.q, 1024, 1024, 1, 0.01, 10)

    def _load_actors(self):
        # here you add various objects (called actors). If your task was to push a ball, you may add a dynamic sphere object on the ground
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        builder = self._scene.create_actor_builder()

        rot_correction = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0))
        builder.add_nonconvex_collision_from_file(
            osp.join(osp.dirname(__file__), "assets/board_1.glb"), rot_correction
        )
        builder.add_visual_from_file(
            osp.join(osp.dirname(__file__), "assets/board_1.glb"), rot_correction
        )
        self.board = builder.build_kinematic("board")
        # build_cube(self._scene, half_size=0.05, color=[0, 1, 0, 1], name="cube", body_type="kinematic")
        builder = self._scene.create_actor_builder()
        builder.add_convex_collision_from_file(
            osp.join(osp.dirname(__file__), "assets/yellow_peg.glb"), rot_correction
        )
        builder.add_visual_from_file(
            osp.join(osp.dirname(__file__), "assets/yellow_peg.glb"), rot_correction
        )
        self.peg = builder.build("yellow_peg")

        builder = self._scene.create_actor_builder()
        builder.add_multiple_convex_collisions_from_file(
            osp.join(osp.dirname(__file__), "assets/purple_u.ply")
        )
        builder.add_visual_from_file(
            osp.join(osp.dirname(__file__), "assets/purple_u.glb"), rot_correction
        )
        self.purple_u = builder.build("purple_u")

        builder = self._scene.create_actor_builder()
        builder.add_multiple_convex_collisions_from_file(
            osp.join(osp.dirname(__file__), "assets/blue_u.ply")
        )
        builder.add_visual_from_file(
            osp.join(osp.dirname(__file__), "assets/blue_u.glb"), rot_correction
        )
        self.blue_u = builder.build("blue_u")

        builder = self._scene.create_actor_builder()
        builder.add_multiple_convex_collisions_from_file(
            osp.join(osp.dirname(__file__), "assets/green_bridge.ply")
        )
        builder.add_visual_from_file(
            osp.join(osp.dirname(__file__), "assets/green_bridge.glb"), rot_correction
        )
        self.bridge = builder.build("green_bridge")

        rot_correction = sapien.Pose(q=euler2quat(np.pi / 2, 0, np.pi / 2))
        builder = self._scene.create_actor_builder()
        builder.add_nonconvex_collision_from_file(
            osp.join(osp.dirname(__file__), "assets/reorienting_fixture.glb"),
            rot_correction,
        )
        builder.add_visual_from_file(
            osp.join(osp.dirname(__file__), "assets/reorienting_fixture.glb"),
            rot_correction,
        )
        self.reorienting_fixture = builder.build_kinematic("reorienting_fixture")

        builder = self._scene.create_actor_builder()
        # builder.add_box_visual(
        #     half_size=[0.02, 0.035, 0.4],
        #     material=sapien.render.RenderMaterial(base_color=[0, 1, 1, 0.3]),
        # )
        self.bridge_grasp = builder.build_kinematic(name="bridge_grasp")

    """
    Episode Initialization Code

    below are all functions involved in episode initialization during environment reset called in the same order. As a user
    you can change these however you want for your desired task.
    """

    def _initialize_actors(self):
        self.table_scene.initialize()
        offset_pose = sapien.Pose(p=[0.05, -0.115, 0], q=euler2quat(0, 0, np.pi / 2))
        self.board.set_pose(
            sapien.Pose(p=np.array([0.115, 0.115, 0.034444])) * offset_pose
        )
        self.peg.set_pose(sapien.Pose(p=np.array([0.115, 0.115, 0.0585])) * offset_pose)
        self.purple_u.set_pose(
            sapien.Pose(p=np.array([0.115, 0.047, 0.06375])) * offset_pose
        )
        self.blue_u.set_pose(
            sapien.Pose(p=np.array([0.115, 0.183, 0.06375])) * offset_pose
        )
        self.bridge.set_pose(
            sapien.Pose(p=np.array([0.115, 0.115, 0.048667])) * offset_pose
        )
        self.reorienting_fixture.set_pose(sapien.Pose(p=np.array([0.05, 0.25, 0.0285])))

        place_order = [self.purple_u, self.blue_u, self.purple_u, self.peg, self.bridge]
        for i, obj in enumerate(place_order):
            obj.set_pose(obj.pose * sapien.Pose([0, 0, 0.03 * i]))

    def _initialize_task(self):
        # we highly recommend to generate some kind of "goal" information to then later include in observations
        # goal can be parameterized as a state (e.g. target pose of a object)
        self.bridge.set_pose(
            sapien.Pose(
                p=[-0.12, 0.23, 0.048667 / 2], q=euler2quat(0, -np.pi / 2, np.pi / 2)
            )
        )
        self.bridge_grasp_offset = sapien.Pose(p=[0, 0, 0.03])
        self.bridge_grasp.set_pose(self.bridge.pose * self.bridge_grasp_offset)

    """
    Modifying observations, goal parameterization, and success conditions for your task

    the code below all impact some part of `self.step` function
    """

    def _get_obs_extra(self):
        # should return an OrderedDict of additional observation data for your tasks
        # this will be included as part of the observation in the "extra" key
        return OrderedDict(
            bridge=vectorize_pose(self.bridge.pose),
            board=vectorize_pose(self.board.pose),
            peg=vectorize_pose(self.peg.pose),
        )

    def evaluate(self, obs: Any):
        self.bridge_grasp.set_pose(self.bridge.pose * self.bridge_grasp_offset)
        # should return a dictionary containing "success": bool indicating if the environment is in success state or not. The value here is also what the sparse reward is
        # for the task. You may also include additional keys which will populate the info object returned by self.step
        return {"success": [False]}

    def compute_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        # you can optionally provide a dense reward function by returning a scalar value here. This is used when reward_mode="dense"
        reward = torch.zeros(self.num_envs, device=self.device)
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
