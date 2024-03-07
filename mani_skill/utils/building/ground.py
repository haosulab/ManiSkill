"""
Useful utilities for creating the ground of a scene
"""
from __future__ import annotations

import os.path as osp
from typing import TYPE_CHECKING

import sapien
import sapien.render

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene


def build_ground(
    scene: ManiSkillScene,
    floor_width=20,
    altitude=0,
    name="ground",
    return_builder=False,
):
    ground = scene.create_actor_builder()
    ground.add_visual_from_file(
        osp.join(osp.dirname(__file__), "assets/floor_tiles_06_2k.glb"),
        pose=sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0.7071068, 0, 0]),
        scale=[floor_width, 1, floor_width],
    )
    ground.add_plane_collision(
        sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0, -0.7071068, 0]),
    )
    if return_builder:
        return ground
    return ground.build_static(name=name)


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
