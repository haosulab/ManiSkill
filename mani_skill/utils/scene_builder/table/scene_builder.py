import os.path as osp
from pathlib import Path
from typing import List

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder


class TableSceneBuilder(SceneBuilder):
    """A simple scene builder that adds a table to the scene such that the height of the table is at 0, and
    gives reasonable initial poses for robots."""

    def __init__(self, env, robot_init_qpos_noise=0.02, custom_table=False):
        super().__init__(env, robot_init_qpos_noise)
        self.custom_table = custom_table

    def _build_custom_table(self, length: float, width: float, height: float):
        """
        Build a custom table with specified dimensions.
        
        Args:
            length: Length of the table (x-axis)
            width: Width of the table (y-axis) 
            height: Height of the table (z-axis)
            
        Returns:
            The built table actor
        """
        # Create actor builder for collision and visual
        builder = self.scene.create_actor_builder()
        
        # Add box collision for the entire table (tabletop + legs)
        table_half_size = [width/2, length/2, height/2]  # half dimensions
        table_pose = sapien.Pose(p=[0, 0, height/2])  # center of the table
        builder.add_box_collision(table_pose, table_half_size)
        
        # Tabletop (black)
        tabletop_material = sapien.render.RenderMaterial(base_color=[0.1, 0.1, 0.1, 1.0])
        tabletop_thickness = 0.05
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, height - tabletop_thickness/2]),
            half_size=[width/2, length/2, tabletop_thickness/2],
            material=tabletop_material
        )
        
        # Table legs (light gray)
        leg_material = sapien.render.RenderMaterial(base_color=[0.9, 0.9, 0.9, 1.0])
        leg_height = height - tabletop_thickness/2
        leg_margin = 0.03  # margin from corners
        leg_size = 0.05  # square legs
        
        leg_positions = [
            [width/2 - leg_margin, length/2 - leg_margin, leg_height/2],   # front right
            [width/2 - leg_margin, -length/2 + leg_margin, leg_height/2],  # front left
            [-width/2 + leg_margin, length/2 - leg_margin, leg_height/2],  # back right
            [-width/2 + leg_margin, -length/2 + leg_margin, leg_height/2], # back left
        ]
        
        for leg_pos in leg_positions:
            builder.add_box_visual(
                pose=sapien.Pose(p=leg_pos),
                half_size=[leg_size/2, leg_size/2, leg_height/2],
                material=leg_material
            )
        
        # Build the final table actor
        return builder.build_kinematic(name="table-custom")
    
    def _build_custom_wall(self, table_pose: sapien.Pose, length: float):
        wall_size = [0.2, 10.0, 4.0]
        wall_half = [s / 2 for s in wall_size]

        wall_offset = 0.25
        table_back_x = table_pose.p[0] - length / 2
        wall_x = table_back_x - wall_half[0] - wall_offset
        wall_pose = sapien.Pose(p=[wall_x, 0, 0])

        white_mat = sapien.render.RenderMaterial(
            base_color=[1, 1, 1, 1], roughness=0.9, metallic=0.0
        )

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(pose=sapien.Pose(), half_size=wall_half)
        builder.add_box_visual   (pose=sapien.Pose(), half_size=wall_half, material=white_mat)

        builder.initial_pose = wall_pose
        wall_actor = builder.build_static(name="white_wall")
        
        return wall_actor

    def build(self):
        if self.custom_table:
            # Use custom table with specified dimensions - height of the glb table, length and width matching real table
            self.table_height = 0.91964292762787
            self.table = self._build_custom_table(length=1.52, width=0.76, height=self.table_height)
            table_pose_world = sapien.Pose(p=[0, 0, self.table_height/2])
            self.wall  = self._build_custom_wall(table_pose_world, length=1.52)

        else:
            # Use default GLB table
            builder = self.scene.create_actor_builder()
            model_dir = Path(osp.dirname(__file__)) / "assets"
            table_model_file = str(model_dir / "table.glb")
            scale = 1.75

            table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
            # builder.add_nonconvex_collision_from_file(
            #     filename=table_model_file,
            #     scale=[scale] * 3,
            #     pose=table_pose,
            # )
            builder.add_box_collision(
                pose=sapien.Pose(p=[0, 0, 0.9196429 / 2]),
                half_size=(2.418 / 2, 1.209 / 2, 0.9196429 / 2),
            )
            builder.add_visual_from_file(
                filename=table_model_file, scale=[scale] * 3, pose=table_pose
            )
            builder.initial_pose = sapien.Pose(
                p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
            )
            self.table = builder.build_kinematic(name="table-workspace")
            # aabb = (
            #     table._objs[0]
            #     .find_component_by_type(sapien.render.RenderBodyComponent)
            #     .compute_global_aabb_tight()
            # )
            # value of the call above is saved below
            aabb = np.array(
                [
                    [-0.7402168, -1.2148621, -0.91964257],
                    [0.4688596, 1.2030163, 3.5762787e-07],
                ]
            )
            self.table_length = aabb[1, 0] - aabb[0, 0]
            self.table_width = aabb[1, 1] - aabb[0, 1]
            self.table_height = aabb[1, 2] - aabb[0, 2]
        
        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500
        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=-self.table_height
        )
        if self.custom_table:
            self.scene_objects: List[sapien.Entity] = [self.table, self.wall, self.ground]
        else:
            self.scene_objects: List[sapien.Entity] = [self.table, self.ground]

    def initialize(self, env_idx: torch.Tensor):
        # table_height = 0.9196429
        b = len(env_idx)
        if self.custom_table:
            self.table.set_pose(sapien.Pose(p=[-0.2245382, 0, -self.table_height]))
        else:
            self.table.set_pose(
                sapien.Pose(p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))
            )
        if self.env.robot_uids == "panda":
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids == "panda_wristcam":
            # fmt: off
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, -np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids in [
            "xarm6_allegro_left",
            "xarm6_allegro_right",
            "xarm6_robotiq",
            "xarm6_nogripper",
            "xarm6_pandagripper",
        ]:
            qpos = self.env.agent.keyframes["rest"].qpos
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.522, 0, 0]))
        elif self.env.robot_uids == "floating_robotiq_2f_85_gripper":
            qpos = self.env.agent.keyframes["open_facing_side"].qpos
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.5, 0, 0.05]))
        elif self.env.robot_uids == "fetch":
            qpos = np.array(
                [
                    0,
                    0,
                    0,
                    0.386,
                    0,
                    0,
                    0,
                    -np.pi / 4,
                    0,
                    np.pi / 4,
                    0,
                    np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-1.05, 0, -self.table_height]))

            self.ground.set_collision_group_bit(
                group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
            )
        elif self.env.robot_uids == ("panda", "panda"):
            agent: MultiAgent = self.env.agent
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(
                sapien.Pose([0, 0.75, 0], q=euler2quat(0, 0, -np.pi / 2))
            )
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(
                sapien.Pose([0, -0.75, 0], q=euler2quat(0, 0, np.pi / 2))
            )
        elif self.env.robot_uids == ("panda_wristcam", "panda_wristcam"):
            agent: MultiAgent = self.env.agent
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(
                sapien.Pose([0, 0.75, 0], q=euler2quat(0, 0, -np.pi / 2))
            )
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(
                sapien.Pose([0, -0.75, 0], q=euler2quat(0, 0, np.pi / 2))
            )
        elif (
            "dclaw" in self.env.robot_uids
            or "allegro" in self.env.robot_uids
            or "trifinger" in self.env.robot_uids
        ):
            # Need to specify the robot qpos for each sub-scenes using tensor api
            pass
        elif self.env.robot_uids == "panda_stick":
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids in ["widowxai", "widowxai_wristcam"]:
            qpos = self.env.agent.keyframes["ready_to_grasp"].qpos
            self.env.agent.reset(qpos)
        elif self.env.robot_uids == "so100":
            qpos = np.array([0, np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, 1.0])
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(
                sapien.Pose([-0.725, 0, 0], q=euler2quat(0, 0, np.pi / 2))
            )
