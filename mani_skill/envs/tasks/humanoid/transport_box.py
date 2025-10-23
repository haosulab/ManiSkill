import copy
import os
from pathlib import Path
from typing import Any

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
from mani_skill.utils.building import ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig


@register_env("UnitreeG1TransportBox-v1", max_episode_steps=100)
class TransportBoxEnv(BaseEnv):
    """
    **Task Description:**
    A G1 humanoid robot must find a box on a table and transport it to the other table and place it there.

    **Randomizations:**
    - the box's xy position is randomized in the region [-0.05, -0.05] x [0.2, 0.05]
    - the box's z-axis rotation is randomized to a random angle in [0, np.pi/6]

    **Success Conditions:**
    - the box is resting on top of the other table
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/UnitreeG1TransportBox-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["unitree_g1_simplified_upper_body_with_head_camera"]
    agent: UnitreeG1UpperBodyWithHeadCamera

    def __init__(self, *args, **kwargs):
        self.init_robot_pose = copy.deepcopy(
            UnitreeG1UpperBodyWithHeadCamera.keyframes["standing"].pose
        )
        self.init_robot_pose.p = [-0.1, 0, 0.755]
        self.init_robot_qpos = UnitreeG1UpperBodyWithHeadCamera.keyframes[
            "standing"
        ].qpos.copy()
        self.init_robot_qpos[4] = -1.25
        self.init_robot_qpos[3] = 1.25
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
            scene_config=SceneConfig(contact_offset=0.02),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([1.0, 0.0, 1.6], [0, 0.0, 0.65])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 3)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.0, 0.0, 1.6], [0, 0.0, 0.65])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=np.pi / 3
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1]))

    def _load_scene(self, options: dict):
        self.ground = ground.build_ground(self.scene, mipmap_levels=7)
        # build two tables

        model_dir = Path(
            os.path.join(
                os.path.dirname(__file__), "../../../utils/scene_builder/table/assets"
            )
        )
        table_model_file = str(model_dir / "table.glb")
        scale = 1.2
        table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(
            filename=table_model_file,
            scale=[scale] * 3,
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        )
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0.630612274 / 2]),
            half_size=(1.658057143 / 2, 0.829028571 / 2, 0.630612274 / 2),
        )
        builder.add_visual_from_file(
            filename=table_model_file, scale=[scale] * 3, pose=table_pose
        )
        builder.initial_pose = sapien.Pose(p=[0, 0.66, 0])
        self.table_1 = builder.build_static(name="table-1")
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(
            filename=table_model_file,
            scale=[scale] * 3,
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        )
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0.630612274 / 2]),
            half_size=(1.658057143 / 2, 0.829028571 / 2, 0.630612274 / 2),
        )
        builder.add_visual_from_file(
            filename=table_model_file, scale=[scale] * 3, pose=table_pose
        )
        builder.initial_pose = sapien.Pose(p=[0, -0.66, 0])
        self.table_2 = builder.build_static(name="table-2")

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=(0.18, 0.12, 0.12), density=200)
        visual_file = os.path.join(
            os.path.dirname(__file__), "assets/cardboard_box/textured.obj"
        )
        builder.add_visual_from_file(
            filename=visual_file,
            scale=[0.12] * 3,
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        )
        builder.initial_pose = sapien.Pose(p=[-0.1, -0.37, 0.7508])
        self.box = builder.build(name="box")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.agent.robot.set_qpos(self.init_robot_qpos)
            self.agent.robot.set_pose(self.init_robot_pose)
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.7508
            xyz[:, 0] = randomization.uniform(-0.05, 0.2, size=(b,))
            xyz[:, 1] = randomization.uniform(-0.05, 0.05, size=(b,))
            xyz[:, :2] += torch.tensor([-0.1, -0.37])
            quat = randomization.random_quaternions(
                n=b, device=self.device, lock_x=True, lock_y=True, bounds=(0, np.pi / 6)
            )
            self.box.set_pose(Pose.create_from_pq(xyz, quat))

    def evaluate(self):
        # left_hand_grasped_box = self.agent.left_hand_is_grasping(self.box, max_angle=110)
        # right_hand_grasped_box = self.agent.right_hand_is_grasping(self.box, max_angle=110)
        l_contact_forces = (
            (
                self.scene.get_pairwise_contact_forces(
                    self.agent.robot.links_map["left_five_link"], self.box
                )
                + self.scene.get_pairwise_contact_forces(
                    self.agent.robot.links_map["left_three_link"], self.box
                )
                + self.scene.get_pairwise_contact_forces(
                    self.agent.robot.links_map["left_palm_link"], self.box
                )
            )
            .abs()
            .sum(dim=1)
        )
        r_contact_forces = (
            (
                self.scene.get_pairwise_contact_forces(
                    self.agent.robot.links_map["right_five_link"], self.box
                )
                + self.scene.get_pairwise_contact_forces(
                    self.agent.robot.links_map["right_three_link"], self.box
                )
                + self.scene.get_pairwise_contact_forces(
                    self.agent.robot.links_map["right_palm_link"], self.box
                )
            )
            .abs()
            .sum(dim=1)
        )
        left_hand_hit_box = l_contact_forces > 10
        right_hand_hit_box = r_contact_forces > 10
        # is grasping the box if both hands contact the box and the tcp of the hands are below the grasp points on the box.
        box_grasped = (
            left_hand_hit_box
            & right_hand_hit_box
            & (
                self.agent.right_tcp.pose.p[:, 2]
                < self.box_right_grasp_point.p[:, 2] + 0.04
            )
            & (
                self.agent.left_tcp.pose.p[:, 2]
                < self.box_left_grasp_point.p[:, 2] + 0.04
            )
        )

        # simply requires box to be resting somewhere on the correct table
        box_at_correct_table_z = (0.751 > self.box.pose.p[:, 2]) & (
            self.box.pose.p[:, 2] > 0.750
        )
        box_at_correct_table_xy = (
            (0.78 > self.box.pose.p[:, 0])
            & (self.box.pose.p[:, 0] > -0.78)
            & (1.0 > self.box.pose.p[:, 1])
            & (self.box.pose.p[:, 1] > 0.3)
        )
        # box_at_correct_table = torch.linalg.norm(self.box.pose.p - torch.tensor([0, 0.66, 0.731], device=self.device), dim=1) < 0.05
        box_at_correct_table = box_at_correct_table_z & box_at_correct_table_xy

        facing_table_with_box = (-1.7 < self.agent.robot.qpos[:, 0]) & (
            self.agent.robot.qpos[:, 0] < -1.4
        )  # in this range the robot is probably facing the box on the left table.
        return {
            "success": ~box_grasped & box_at_correct_table,
            "left_hand_hit_box": l_contact_forces > 0,
            "right_hand_hit_box": r_contact_forces > 0,
            "box_grasped": box_grasped,
            "box_at_correct_table_xy": box_at_correct_table_xy,
            "facing_table_with_box": facing_table_with_box,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            right_tcp_pose=self.agent.right_tcp.pose.raw_pose,
            left_tcp_pose=self.agent.left_tcp.pose.raw_pose,
        )

        if "state" in self.obs_mode:
            obs.update(
                box_pose=self.box.pose.raw_pose,
                right_tcp_to_box_pos=self.box.pose.p - self.agent.right_tcp.pose.p,
                left_tcp_to_box_pos=self.box.pose.p - self.agent.left_tcp.pose.p,
            )
        return obs

    @property
    def box_right_grasp_point(self):
        return self.box.pose * Pose.create_from_pq(
            torch.tensor([-0.165, 0.07, 0.05], device=self.device)
        )

    @property
    def box_left_grasp_point(self):
        return self.box.pose * Pose.create_from_pq(
            torch.tensor([0.165, 0.07, 0.05], device=self.device)
        )

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Stage 1, move to face the box on the table. Succeeds if facing_table_with_box
        reward = 1 - torch.tanh((self.agent.robot.qpos[:, 0] + 1.4).abs())

        # Stage 2, grasp the box stably. Succeeds if box_grasped
        # encourage arms to go down essentially and for tcps to be close to the edge of the box
        stage_2_reward = (
            1
            + (1 - torch.tanh((self.agent.robot.qpos[:, 3]).abs())) / 4
            + (1 - torch.tanh((self.agent.robot.qpos[:, 4]).abs())) / 4
            + (
                1
                - torch.tanh(
                    3
                    * torch.linalg.norm(
                        self.agent.right_tcp.pose.p - self.box_right_grasp_point.p,
                        dim=1,
                    )
                )
            )
            / 4
            + (
                1
                - torch.tanh(
                    3
                    * torch.linalg.norm(
                        self.agent.left_tcp.pose.p - self.box_left_grasp_point.p, dim=1
                    )
                )
            )
            / 4
        )
        reward[info["facing_table_with_box"]] = stage_2_reward[
            info["facing_table_with_box"]
        ]
        # Stage 3 transport box to above the other table, Succeeds if box_at_correct_table_xy
        stage_3_reward = (
            2 + 1 - torch.tanh((self.agent.robot.qpos[:, 0] - 1.4).abs() / 5)
        )
        reward[info["box_grasped"]] = stage_3_reward[info["box_grasped"]]
        # Stage 4 let go of the box. Succeeds if success (~box_grasped & box_at_correct_table)
        stage_4_reward = (
            3
            + (1 - torch.tanh((self.agent.robot.qpos[:, 3] - 1.25).abs())) / 2
            + (1 - torch.tanh((self.agent.robot.qpos[:, 4] + 1.25).abs())) / 2
        )
        reward[info["box_at_correct_table_xy"]] = stage_4_reward[
            info["box_at_correct_table_xy"]
        ]
        # encourage agent to stay close to a target qposition?
        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs, action, info) / 5
