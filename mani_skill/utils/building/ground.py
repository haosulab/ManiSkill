"""
Useful utilities for creating the ground of a scene
"""
from __future__ import annotations

import os.path as osp
from typing import TYPE_CHECKING

import numpy as np
import sapien
import sapien.render

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene


def build_ground(
    scene: ManiSkillScene,
    floor_width: int = 100,
    altitude=0,
    name="ground",
    return_builder=False,
):
    ground = scene.create_actor_builder()
    ground.add_plane_collision(
        sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0, -0.7071068, 0]),
    )
    if return_builder:
        return ground
    actor = ground.build_static(name=name)

    floor_comp = sapien.render.RenderBodyComponent()

    # generate a grid of right triangles that form 1x1 meter squares centered at (0, 0, 0)
    num_verts = (floor_width + 1) ** 2
    vertices = np.zeros((num_verts, 3))
    floor_half_width = floor_width / 2
    ranges = np.arange(start=-floor_half_width, stop=floor_half_width + 1)
    xx, yy = np.meshgrid(ranges, ranges)
    xys = np.stack((yy, xx), axis=2).reshape(-1, 2)
    vertices[:, 0] = xys[:, 0]
    vertices[:, 1] = xys[:, 1]
    normals = np.zeros((len(vertices), 3))
    normals[:, 2] = 1

    mat = sapien.render.RenderMaterial()
    mat.diffuse_texture = sapien.render.RenderTexture2D(
        filename=osp.join(
            osp.dirname(__file__), "assets/floor_tiles_06_diff_2k_aligned.png"
        )
    )
    mat_square_len = 4  # hardcoded for the floor tile picture, saying that square tile is 4 meters wide
    uv_scale = floor_width / mat_square_len
    uvs = np.zeros((len(vertices), 2))
    uvs[:, 0] = (xys[:, 0] * uv_scale + floor_half_width) / floor_width
    uvs[:, 1] = (xys[:, 1] * uv_scale + floor_half_width) / floor_width

    # TODO: This is fast but still two for loops which is a little annoying
    triangles = []
    for i in range(floor_width):
        triangles.append(
            np.stack(
                [
                    np.arange(floor_width) + i * (floor_width + 1),
                    np.arange(floor_width) + 1 + floor_width + i * (floor_width + 1),
                    np.arange(floor_width) + 1 + i * (floor_width + 1),
                ],
                axis=1,
            )
        )
    for i in range(floor_width):
        triangles.append(
            np.stack(
                [
                    np.arange(floor_width) + 1 + floor_width + i * (floor_width + 1),
                    np.arange(floor_width) + floor_width + 2 + i * (floor_width + 1),
                    np.arange(floor_width) + 1 + i * (floor_width + 1),
                ],
                axis=1,
            )
        )
    triangles = np.concatenate(triangles)

    shape = sapien.render.RenderShapeTriangleMesh(
        vertices=vertices, triangles=triangles, normals=normals, uvs=uvs, material=mat
    )
    floor_comp.attach(shape)
    for obj in actor._objs:
        obj.add_component(floor_comp)


def build_meter_ground(
    scene: ManiSkillScene,
    floor_width=20,
    altitude=0,
    name="ground",
    return_builder=False,
):
    ground = scene.create_actor_builder()

    # for i in range(floor_width * 2):
    #     ground.add_box_visual(pose=sapien.Pose(p=[i - floor_width, 0, altitude]), half_size=[0.01, floor_width, 2e-4], material=sapien.render.RenderMaterial(base_color=[0.02, 0.02, 0.04, 1]))
    # for i in range(floor_width * 2):
    #     ground.add_box_visual(pose=sapien.Pose(p=[0, i - floor_width, altitude]), half_size=[floor_width, 0.01, 2e-4], material=sapien.render.RenderMaterial(base_color=[0.02, 0.02, 0.04, 1]))
    # for i in range(floor_width):
    #     ground.add_box_visual(pose=sapien.Pose(p=[i * 2 - floor_width, 0, altitude]), half_size=[0.5, floor_width, 2e-2], material=sapien.render.RenderMaterial(base_color=[0.02, 0.02, 0.04, 1]))
    # for i in range(floor_width):
    #     ground.add_box_visual(pose=sapien.Pose(p=[0, i * 2 - floor_width, altitude]), half_size=[floor_width, 0.5, 2e-2], material=sapien.render.RenderMaterial(base_color=[0.02, 0.02, 0.04, 1]))

    ground.add_plane_collision(
        pose=sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0, -0.7071068, 0]),
    )
    ground.add_plane_visual(
        pose=sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0, -0.7071068, 0]),
        scale=(floor_width, floor_width, floor_width),
        material=sapien.render.RenderMaterial(
            base_color=[0.9, 0.9, 0.93, 1], metallic=0.5, roughness=0.5
        ),
    )
    if return_builder:
        return ground
    return ground.build_static(name=name)
