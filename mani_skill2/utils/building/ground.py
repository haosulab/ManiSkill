"""
Useful utilities for creating the ground of a scene
"""
import os.path as osp

import sapien
import sapien.render


def build_tesselated_square_floor(scene: sapien.Scene, floor_width=20, altitude=0):
    ground = scene.create_actor_builder()
    ground.add_visual_from_file(
        osp.join(osp.dirname(__file__), "assets/floor_tiles_06_2k.glb"),
        pose=sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0.7071068, 0, 0]),
        scale=[floor_width, 1, floor_width],
    )
    ground.add_plane_collision(
        sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0, -0.7071068, 0]),
    )
    return ground.build_static(name="ground")
