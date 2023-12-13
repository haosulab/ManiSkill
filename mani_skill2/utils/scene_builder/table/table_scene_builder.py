import itertools
import os.path as osp
from pathlib import Path
from typing import Dict, List, Type, Union

import numpy as np
import sapien
import sapien.render
from transforms3d.euler import euler2quat

from mani_skill2.utils.building.ground import build_tesselated_square_floor
from mani_skill2.utils.scene_builder import SceneBuilder


class TableSceneBuilder(SceneBuilder):
    def __init__(self):
        super().__init__()

    def build(self, scene: sapien.Scene, **kwargs):
        builder = scene.create_actor_builder()
        model_dir = Path(osp.dirname(__file__)) / "assets"
        scale = 1.2
        physical_material = None

        collision_file = str(model_dir / "Dining_Table_204_1.glb")  # a metal table
        table_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0))
        builder.add_nonconvex_collision_from_file(
            filename=collision_file,
            scale=[scale] * 3,
            material=physical_material,
            pose=table_pose,
        )

        visual_file = str(model_dir / "Dining_Table_204_1.glb")
        builder.add_visual_from_file(
            filename=visual_file, scale=[scale] * 3, pose=table_pose
        )
        table = builder.build_static(name="table-workspace")
        aabb = table.find_component_by_type(
            sapien.render.RenderBodyComponent
        ).compute_global_aabb_tight()
        table_height = aabb[1, 2] - aabb[0, 2]
        table.set_pose(
            sapien.Pose(p=[-0.1, 0, -table_height], q=euler2quat(0, 0, np.pi / 2))
        )

        self.ground = build_tesselated_square_floor(scene, altitude=-table_height)
        self.table = table
        self._scene_objects: List[sapien.Entity] = [self.table, self.ground]

    @property
    def scene_objects(self):
        return self._scene_objects

    @property
    def movable_objects(self):
        raise AttributeError(
            "For TableScene, additional movable objects must be added and managed at Task-level"
        )
