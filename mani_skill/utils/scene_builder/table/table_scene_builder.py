import os.path as osp
from pathlib import Path
from typing import List

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.fetch import FETCH_UNIQUE_COLLISION_BIT
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder


# TODO (stao): make the build and initialize api consistent with other scenes
class TableSceneBuilder(SceneBuilder):
    robot_init_qpos_noise: float = 0.02

    def build(self):
        builder = self.scene.create_actor_builder()
        model_dir = Path(osp.dirname(__file__)) / "assets"
        table_model_file = str(model_dir / "table.glb")
        scale = 1.75

        table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
        builder.add_nonconvex_collision_from_file(
            filename=table_model_file,
            scale=[scale] * 3,
            pose=table_pose,
        )
        builder.add_visual_from_file(
            filename=table_model_file, scale=[scale] * 3, pose=table_pose
        )
        table = builder.build_kinematic(name="table-workspace")
        aabb = (
            table._objs[0]
            .find_component_by_type(sapien.render.RenderBodyComponent)
            .compute_global_aabb_tight()
        )
        self.table_length = aabb[1, 0] - aabb[0, 0]
        self.table_width = aabb[1, 1] - aabb[0, 1]
        self.table_height = aabb[1, 2] - aabb[0, 2]

        self.ground = build_ground(self.scene, altitude=-self.table_height)
        self.table = table
        self._scene_objects: List[sapien.Entity] = [self.table, self.ground]

    def initialize(self, env_idx: torch.Tensor):
        # table_height = 0.9196429
        b = len(env_idx)
        self.table.set_pose(
            sapien.Pose(p=[-0.12, 0, -self.table_height], q=euler2quat(0, 0, np.pi / 2))
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
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids == "panda_realsensed435":
            # fmt: off
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, -np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids == "xmate3_robotiq":
            qpos = np.array(
                [0, np.pi / 6, 0, np.pi / 3, 0, np.pi / 2, -np.pi / 2, 0, 0]
            )
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            qpos[:, -2:] = 0
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.562, 0, 0]))
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

            for body in self.ground._bodies:
                for cs in body.get_collision_shapes():
                    cg = cs.get_collision_groups()
                    cg[2] |= FETCH_UNIQUE_COLLISION_BIT
                    cs.set_collision_groups(cg)
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
        elif "dclaw" in self.env.robot_uids or "allegro" in self.env.robot_uids:
            # Need to specify the robot qpos for each sub-scenes using tensor api
            pass

    @property
    def scene_objects(self):
        return self._scene_objects

    @property
    def movable_objects(self):
        raise AttributeError(
            "For TableScene, additional movable objects must be added and managed at Task-level"
        )
