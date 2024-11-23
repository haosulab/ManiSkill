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


class KitchenCounterSceneBuilder(SceneBuilder):
    def build(self, scale=1.0):
        builder = self.scene.create_actor_builder()
        model_dir = Path(osp.dirname(__file__)) / "assets"
        table_model_file = str(model_dir / "kitchen_counter.glb")
        table_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, np.pi))
        builder.add_nonconvex_collision_from_file(
            filename=table_model_file, pose=table_pose, scale=[scale] * 3
        )
        builder.add_visual_from_file(
            filename=table_model_file, scale=[scale] * 3, pose=table_pose
        )
        builder.initial_pose = sapien.Pose()
        table = builder.build_static(name="kitchen-counter")

        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500
        self.ground = build_ground(self.scene, floor_width=floor_width, altitude=0)
        self.table = table
        self.scene_objects: List[sapien.Entity] = [self.table, self.ground]

    def initialize(self, env_idx: torch.Tensor):
        pass
