"""
Useful utilities for creating the ground of a scene
"""
import os.path as osp

import sapien
import sapien.render


def build_tesselated_square_floor(scene: sapien.Scene, floor_width=20, altitude=0):
    ground = scene.create_actor_builder()
    rend_mtl = sapien.render.RenderMaterial(
        base_color=[0.06, 0.08, 0.12, 1],
        metallic=0.0,
        roughness=0.9,
        specular=0.8,
    )
    rend_mtl.diffuse_texture = sapien.render.RenderTexture2D(
        osp.join(osp.dirname(__file__), "assets/floor_texture_4.png")
    )
    ground.add_visual_from_file(
        osp.join(osp.dirname(__file__), "assets/tiled_floor.obj"),
        pose=sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0.7071068, 0, 0]),
        scale=[floor_width, 1, floor_width],
        material=rend_mtl,
    )
    ground.add_plane_collision(
        sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0, -0.7071068, 0]),
    )
    return ground.build_static(name="ground")
