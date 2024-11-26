import copy
import os
from pathlib import Path
from typing import List

import numpy as np
import sapien
from torch import Tensor
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.unitree_g1.g1_upper_body import UnitreeG1UpperBody
from mani_skill.utils.building import ground
from mani_skill.utils.scene_builder.scene_builder import SceneBuilder


class TableSceneBuilder(SceneBuilder):
    def build(self, build_config_idxs: List[int] = None):
        self.ground = ground.build_ground(self.scene, mipmap_levels=7)
        # build two tables

        model_dir = Path(
            os.path.join(
                os.path.dirname(__file__),
                "../../../../utils/scene_builder/table/assets",
            )
        )
        table_model_file = str(model_dir / "table.glb")
        scale = 1.2
        # table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0.630612274 / 2]),
            half_size=(1.658057143 / 2, 0.829028571 / 2, 0.630612274 / 2),
        )
        builder.add_visual_from_file(filename=table_model_file, scale=[scale] * 3)
        builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        self.table = builder.build_static(name="table")
        if (
            self.env.robot_uids == "unitree_g1_simplified_upper_body_with_head_camera"
            or self.env.robot_uids == "unitree_g1_simplified_upper_body"
        ):
            self.robot_init_pose = copy.deepcopy(
                UnitreeG1UpperBody.keyframes["standing"].pose
            )
            self.robot_init_qpos = copy.deepcopy(
                UnitreeG1UpperBody.keyframes["standing"].qpos
            )
            self.robot_init_pose.p = [-0.1, 0, 0.755]

    def initialize(self, env_idx: Tensor, init_config_idxs: List[int] = None):
        pass
