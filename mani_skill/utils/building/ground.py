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
    floor_length: int = None,
    xy_origin: tuple = (0, 0),
    altitude=0,
    name="ground",
    texture_file=osp.join(osp.dirname(__file__), "assets/grid_texture.png"),
    texture_square_len=4,
    mipmap_levels=4,
    add_collision=True,
):
    """Procedurally creates a checkered floor given a floor width in meters.

    Note that this function runs slower as floor width becomes larger, but in general this function takes no more than 0.05s to run
    and usually is never run more than once as it is for building a scene, not loading.
    """
    ground = scene.create_actor_builder()
    if add_collision:
        ground.add_plane_collision(
            sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0, -0.7071068, 0]),
        )
    ground.initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
    if scene.parallel_in_single_scene:
        # when building a ground and using a parallel render in the GUI, we want to only build one ground visual+collision plane
        ground.set_scene_idxs([0])
    actor = ground.build_static(name=name)

    # generate a grid of right triangles that form 1x1 meter squares centered at (0, 0, 0)
    floor_length = floor_width if floor_length is None else floor_length
    num_verts = (floor_width + 1) * (floor_length + 1)
    vertices = np.zeros((num_verts, 3))
    floor_half_width = floor_width / 2
    floor_half_length = floor_length / 2
    xrange = np.arange(start=-floor_half_width, stop=floor_half_width + 1)
    yrange = np.arange(start=-floor_half_length, stop=floor_half_length + 1)
    xx, yy = np.meshgrid(xrange, yrange)
    xys = np.stack((yy, xx), axis=2).reshape(-1, 2)
    vertices[:, 0] = xys[:, 0] + xy_origin[0]
    vertices[:, 1] = xys[:, 1] + xy_origin[1]
    vertices[:, 2] = altitude
    normals = np.zeros((len(vertices), 3))
    normals[:, 2] = 1

    mat = sapien.render.RenderMaterial()
    mat.base_color_texture = sapien.render.RenderTexture2D(
        filename=texture_file,
        mipmap_levels=mipmap_levels,
    )
    uv_scale = floor_width / texture_square_len
    uvs = np.zeros((len(vertices), 2))
    uvs[:, 0] = (xys[:, 0] * uv_scale + floor_half_width) / floor_width
    uvs[:, 1] = (xys[:, 1] * uv_scale + floor_half_width) / floor_width

    # TODO: This is fast but still two for loops which is a little annoying
    triangles = []
    for i in range(floor_length):
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
    for i in range(floor_length):
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
