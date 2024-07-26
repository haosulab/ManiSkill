from typing import List

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder


class PlanarSceneBuilder(SceneBuilder):
    def build(self, build_config_idxs: List[int] = None):
        # ground - a strip with length along +x
        self.ground = build_ground(
            self.scene,
            floor_width=2,
            floor_length=100,
            altitude=0,
            xy_origin=(50 - 2, 0),
        )

        # background visual wall
        self.wall = self.scene.create_actor_builder()
        self.wall.add_box_visual(
            half_size=(1e-3, 65, 10),
            pose=sapien.Pose(p=[(50 - 2), 2, 0], q=euler2quat(0, 0, np.pi / 2)),
            material=sapien.render.RenderMaterial(
                base_color=np.array([0.3, 0.3, 0.3, 1])
            ),
        )
        self.wall.build_static(name="wall")
        self.scene_objects: List[sapien.Entity] = [self.ground, self.wall]
