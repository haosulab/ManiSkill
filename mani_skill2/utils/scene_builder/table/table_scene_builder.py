import os.path as osp
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import sapien
import sapien.render
from transforms3d.euler import euler2quat

from mani_skill2.utils.building.ground import build_tesselated_square_floor
from mani_skill2.utils.scene_builder import SceneBuilder


@dataclass
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
        self.table_height = aabb[1, 2] - aabb[0, 2]

        self.ground = build_tesselated_square_floor(
            self.scene, altitude=-self.table_height
        )
        self.table = table
        self._scene_objects: List[sapien.Entity] = [self.table, self.ground]

    def initialize(self):
        # table_height = 0.9196429
        self.table.set_pose(
            sapien.Pose(p=[-0.12, 0, -self.table_height], q=euler2quat(0, 0, np.pi / 2))
        )
        if self.env.robot_uid == "panda":
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
            qpos[:-2] += self.env._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uid == "xmate3_robotiq":
            qpos = np.array(
                [0, np.pi / 6, 0, np.pi / 3, 0, np.pi / 2, -np.pi / 2, 0, 0]
            )
            qpos[:-2] += self.env._episode_rng.normal(
                0, self.env.robot_init_qpos_noise, len(qpos) - 2
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.562, 0, 0]))
        elif self.env.robot_uid == "fetch":
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
            self.env.agent.robot.set_pose(sapien.Pose([-0.82, 0, -self.table_height]))

            from mani_skill2.agents.robots.fetch import FETCH_UNIQUE_COLLISION_BIT

            cs = (
                self.ground._objs[0]
                .find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
                .get_collision_shapes()[0]
            )
            cg = cs.get_collision_groups()
            cg[2] = FETCH_UNIQUE_COLLISION_BIT
            cs.set_collision_groups(cg)
        else:
            raise NotImplementedError(self.env.robot_uid)

    @property
    def scene_objects(self):
        return self._scene_objects

    @property
    def movable_objects(self):
        raise AttributeError(
            "For TableScene, additional movable objects must be added and managed at Task-level"
        )
