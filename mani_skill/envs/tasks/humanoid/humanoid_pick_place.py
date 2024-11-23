import copy
import os
from typing import Any, Dict

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.unitree_g1.g1_upper_body import (
    UnitreeG1UpperBodyWithHeadCamera,
)
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.kitchen_counter import KitchenCounterSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig


class HumanoidPickPlaceEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    """sets up a basic scene with a apple to pick up and place on a dish"""
    kitchen_scene_scale = 1.0

    def __init__(self, *args, robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1]))

    def _load_scene(self, options: dict):
        self.scene_builder = KitchenCounterSceneBuilder(self)
        self.kitchen_scene = self.scene_builder.build(scale=self.kitchen_scene_scale)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.scene_builder.initialize(env_idx)

    def evaluate(self):
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=bool),
            "fail": torch.zeros(self.num_envs, device=self.device, dtype=bool),
        }

    def _get_obs_extra(self, info: Dict):
        return dict()


class HumanoidPlaceAppleInBowl(HumanoidPickPlaceEnv):
    SUPPORTED_REWARD_MODES = ["normalized_dense", "dense", "sparse", "none"]

    @property
    def _default_sensor_configs(self):
        return CameraConfig(
            "base_camera",
            sapien.Pose(
                [0.279123, 0.303438, 1.34794], [0.252428, 0.396735, 0.114442, -0.875091]
            ),
            128,
            128,
            np.pi / 2,
            0.01,
            100,
        )

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig(
            "render_camera",
            sapien.Pose(
                [0.279123, 0.303438, 1.34794], [0.252428, 0.396735, 0.114442, -0.875091]
            ),
            512,
            512,
            np.pi / 2,
            0.01,
            100,
        )

    def _load_scene(self, options: Dict):
        super()._load_scene(options)
        scale = self.kitchen_scene_scale
        builder = self.scene.create_actor_builder()
        fix_rotation_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0))
        model_dir = os.path.dirname(__file__) + "/assets"
        builder.add_nonconvex_collision_from_file(
            filename=os.path.join(model_dir, "frl_apartment_bowl_07.ply"),
            pose=fix_rotation_pose,
            scale=[scale] * 3,
        )
        builder.add_visual_from_file(
            filename=os.path.join(model_dir, "frl_apartment_bowl_07.glb"),
            scale=[scale] * 3,
            pose=fix_rotation_pose,
        )
        builder.initial_pose = sapien.Pose(p=[0, -0.4, 0.753])
        self.bowl = builder.build_kinematic(name="bowl")

        builder = self.scene.create_actor_builder()
        model_dir = os.path.dirname(__file__) + "/assets"
        builder.add_multiple_convex_collisions_from_file(
            filename=os.path.join(model_dir, "apple_1.ply"),
            pose=fix_rotation_pose,
            scale=[scale * 0.8]
            * 3,  # scale down more to make apple a bit smaller to be graspable
        )
        builder.add_visual_from_file(
            filename=os.path.join(model_dir, "apple_1.glb"),
            scale=[scale * 0.8] * 3,
            pose=fix_rotation_pose,
        )
        builder.initial_pose = sapien.Pose(p=[0, -0.4, 0.78])
        self.apple = builder.build(name="apple")

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.bowl.pose.p - self.apple.pose.p, axis=1) <= 0.05
        )
        hand_outside_bowl = (
            self.agent.right_tcp.pose.p[:, 2] > self.bowl.pose.p[:, 2] + 0.125
        )
        is_grasped = self.agent.right_hand_is_grasping(self.apple, max_angle=110)
        return {
            "success": is_obj_placed & hand_outside_bowl,
            "hand_outside_bowl": hand_outside_bowl,
            "is_grasped": is_grasped,
        }

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.right_tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                bowl_pos=self.bowl.pose.p,
                obj_pose=self.apple.pose.raw_pose,
                tcp_to_obj_pos=self.apple.pose.p - self.agent.right_tcp.pose.p,
                obj_to_goal_pos=self.bowl.pose.p - self.apple.pose.p,
            )
        return obs

    def _grasp_release_reward(self):
        """a dense reward that rewards the agent for opening their hand"""
        return 1 - torch.tanh(self.agent.right_hand_dist_to_open_grasp())

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.apple.pose.p - self.agent.right_tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        # encourage to bring apple to above the bowl then drop it.
        obj_to_goal_dist = torch.linalg.norm(
            (self.bowl.pose.p + torch.tensor([0, 0, 0.15], device=self.device))
            - self.apple.pose.p,
            axis=1,
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        # once above the goal, encourage to have the hand above the bowl still and begin releasing the grasp
        obj_high_above_bowl = obj_to_goal_dist < 0.025
        grasp_release_reward = self._grasp_release_reward()
        reward[obj_high_above_bowl] = (
            4
            + place_reward[obj_high_above_bowl]
            + grasp_release_reward[obj_high_above_bowl]
        )
        reward[info["success"]] = (
            8 + (place_reward + grasp_release_reward)[info["success"]]
        )
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10


@register_env("UnitreeG1PlaceAppleInBowl-v1", max_episode_steps=100)
class UnitreeG1PlaceAppleInBowlEnv(HumanoidPlaceAppleInBowl):
    """
    **Task Description:**
    Control the humanoid unitree G1 robot to grab an apple with its right arm and place it in a bowl to the side

    **Randomizations:**
    - the bowl's xy position is randomized on top of a table in the region [0.025, 0.025] x [-0.025, -0.025]. It is placed flat on the table
    - the apple's xy position is randomized on top of a table in the region [0.025, 0.025] x [-0.025, -0.025]. It is placed flat on the table
    - the apple's z-axis rotation is randomized to a random angle

    **Success Conditions:**
    - the apple position is within 0.05m euclidean distance of the bowl's position.
    - the robot's right hand is kept outside the bowl and is above it by at least 0.125m.

    **Goal Specification:**
    - The bowl's 3D position
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/UnitreeG1PlaceAppleInBowl-v1_rt.mp4"

    SUPPORTED_ROBOTS = ["unitree_g1_simplified_upper_body_with_head_camera"]
    agent: UnitreeG1UpperBodyWithHeadCamera
    kitchen_scene_scale = 0.82

    def __init__(self, *args, **kwargs):
        self.init_robot_pose = copy.deepcopy(
            UnitreeG1UpperBodyWithHeadCamera.keyframes["standing"].pose
        )
        self.init_robot_pose.p = [-0.3, 0, 0.755]
        super().__init__(
            *args,
            robot_uids="unitree_g1_simplified_upper_body_with_head_camera",
            **kwargs
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22, max_rigid_patch_count=2**21
            ),
            # TODO (stao): G1 robot may need some custom collision disabling as the dextrous fingers may often be close to each other
            # and slow down simulation. A temporary fix is to reduce contact_offset value down so that we don't check so many possible
            # collisions
            scene_config=SceneConfig(contact_offset=0.01),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            # initialize the robot
            self.agent.robot.set_qpos(self.agent.keyframes["standing"].qpos)
            self.agent.robot.set_pose(self.init_robot_pose)

            # initialize the apple to be within reach
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = randomization.uniform(low=-0.025, high=0.025, size=(b, 2))
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            xyz[:, 2] = 0.7335
            self.apple.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.zeros((b, 3))
            xyz[:, :2] = randomization.uniform(low=-0.025, high=0.025, size=(b, 2))
            xyz[:, :2] += torch.tensor([0.0, -0.4])
            xyz[:, 2] = 0.753
            self.bowl.set_pose(Pose.create_from_pq(xyz))
