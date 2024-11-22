import copy
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.unitree_g1.g1_upper_body import UnitreeG1UpperBody
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.kitchen_counter import KitchenCounterSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig


@register_env("UnitreeG1TransportBox-v1", max_episode_steps=100)
class TransportBoxEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["unitree_g1_simplified_upper_body"]
    agent: UnitreeG1UpperBody

    def __init__(self, *args, **kwargs):
        self.init_robot_pose = copy.deepcopy(
            UnitreeG1UpperBody.keyframes["standing"].pose
        )
        self.init_robot_pose.p = [-0.1, 0, 0.755]
        self.init_robot_qpos = UnitreeG1UpperBody.keyframes["standing"].qpos.copy()
        self.init_robot_qpos[4] = -1.25
        self.init_robot_qpos[3] = 1.25
        super().__init__(*args, robot_uids="unitree_g1_simplified_upper_body", **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22, max_rigid_patch_count=2**21
            ),
            scene_config=SceneConfig(contact_offset=0.02),
        )

    # TODO tune cameras
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.53, 0.0, 1.4], [-0.2, 0.0, 0.65])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.53, 0.0, 1.4], [-0.2, 0.0, 0.65])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=np.pi / 2
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1]))

    def _load_scene(self, options: dict):
        self.ground = ground.build_ground(self.scene)
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
        builder.add_box_collision(half_size=(0.18, 0.1, 0.12), density=200)
        builder.add_box_visual(
            half_size=(0.18, 0.1, 0.12),
            material=sapien.render.RenderMaterial(base_color=[0.3, 0.6, 0.2, 1]),
        )
        builder.initial_pose = sapien.Pose(p=[-0.1, -0.37, 0.7508])
        self.box = builder.build(name="box")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            len(env_idx)
            self.agent.robot.set_qpos(self.init_robot_qpos)
            self.agent.robot.set_pose(self.init_robot_pose)
            self.box.set_pose(sapien.Pose(p=[-0.1, -0.37, 0.7508]))

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
        box_grasped = left_hand_hit_box & right_hand_hit_box
        # simply requires box to be resting on the correct table
        box_at_correct_table_z = 0.751 > self.box.pose.p[:, 2] > 0.750
        box_at_correct_table_xy = (0.78 > self.box.pose.p[:, 0] > -0.78) & (
            1.0 > self.box.pose.p[:, 1] > 0.3
        )
        # box_at_correct_table = torch.linalg.norm(self.box.pose.p - torch.tensor([0, 0.66, 0.731], device=self.device), dim=1) < 0.05
        box_at_correct_table = box_at_correct_table_z & box_at_correct_table_xy

        facing_table_with_box = (
            -1.7 < self.agent.robot.qpos[:, 0] < -1.2
        )  # in this range the robot is probably facing the box on the left table.
        return {
            "success": ~box_grasped & box_at_correct_table,
            "left_hand_hit_box": l_contact_forces > 0,
            "right_hand_hit_box": r_contact_forces > 0,
            "box_grasped": box_grasped,
            "box_at_correct_table_xy": box_at_correct_table_xy,
        }

    def _get_obs_extra(self, info: Dict):
        return dict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Stage 1, move to face the box on the table. Succeeds if facing_table_with_box

        # Stage 2, grasp the box stably. Succeeds if box_grasped

        # Stage 3 transport box to above the other table, Succeeds if box_at_correct_table_xy

        # Stage 4 let go of the box. Succeeds if success (~box_grasped & box_at_correct_table)
        pass

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return super().compute_dense_reward(obs, action, info) / 10
