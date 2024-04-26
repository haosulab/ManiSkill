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
):
    """Procedurally creates a checkered floor given a floor width in meters.

    Note that this function runs slower as floor width becomes larger, but in general this function takes no more than 0.05s to run
    and usually is never run more than once as it is for building a scene, not loading.
    """
    ground = scene.create_actor_builder()
    ground.add_plane_collision(
        sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0, -0.7071068, 0]),
    )
    actor = ground.build_static(name=name)

    # generate a grid of right triangles that form 1x1 meter squares centered at (0, 0, 0)
    num_verts = (floor_width + 1) ** 2
    vertices = np.zeros((num_verts, 3))
    floor_half_width = floor_width / 2
    ranges = np.arange(start=-floor_half_width, stop=floor_half_width + 1)
    xx, yy = np.meshgrid(ranges, ranges)
    xys = np.stack((yy, xx), axis=2).reshape(-1, 2)
    vertices[:, 0] = xys[:, 0]
    vertices[:, 1] = xys[:, 1]
    vertices[:, 2] = altitude
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

    for obj in actor._objs:
        floor_comp = sapien.render.RenderBodyComponent()
        floor_comp.attach(shape)
        obj.add_component(floor_comp)

    return actor
