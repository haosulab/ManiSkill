"""
Common utilities for adding primitive prebuilt shapes to a scene
"""

import numpy as np
import sapien
import sapien.render

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building.actor_builder import ActorBuilder


def _build_by_type(builder: ActorBuilder, name, body_type):
    if body_type == "dynamic":
        actor = builder.build(name=name)
    elif body_type == "static":
        actor = builder.build_static(name=name)
    elif body_type == "kinematic":
        actor = builder.build_kinematic(name=name)
    else:
        raise ValueError(f"Unknown body type {body_type}")
    return actor


# Primitive Shapes
def build_cube(
    scene: ManiSkillScene,
    half_size: float,
    color,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
):
    builder = scene.create_actor_builder()
    if add_collision:
        builder.add_box_collision(
            half_size=[half_size] * 3,
        )
    builder.add_box_visual(
        half_size=[half_size] * 3,
        material=sapien.render.RenderMaterial(
            base_color=color,
        ),
    )
    return _build_by_type(builder, name, body_type)


def build_box(
    scene: ManiSkillScene,
    half_sizes,
    color,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
):
    builder = scene.create_actor_builder()
    if add_collision:
        builder.add_box_collision(
            half_size=half_sizes,
        )
    builder.add_box_visual(
        half_size=half_sizes,
        material=sapien.render.RenderMaterial(
            base_color=color,
        ),
    )
    return _build_by_type(builder, name, body_type)


def build_sphere(
    scene: ManiSkillScene,
    radius: float,
    color,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
):
    builder = scene.create_actor_builder()
    if add_collision:
        builder.add_sphere_collision(
            radius=radius,
        )
    builder.add_sphere_visual(
        radius=radius,
        material=sapien.render.RenderMaterial(
            base_color=color,
        ),
    )
    return _build_by_type(builder, name, body_type)


def build_red_white_target(
    scene: ManiSkillScene,
    radius: float,
    thickness: float,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
):
    TARGET_RED = np.array([194, 19, 22, 255]) / 255
    builder = scene.create_actor_builder()
    builder.add_cylinder_visual(
        radius=radius,
        half_length=thickness / 2,
        material=sapien.render.RenderMaterial(base_color=TARGET_RED),
    )
    builder.add_cylinder_visual(
        radius=radius * 4 / 5,
        half_length=thickness / 2 + 1e-5,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
    )
    builder.add_cylinder_visual(
        radius=radius * 3 / 5,
        half_length=thickness / 2 + 2e-5,
        material=sapien.render.RenderMaterial(base_color=TARGET_RED),
    )
    builder.add_cylinder_visual(
        radius=radius * 2 / 5,
        half_length=thickness / 2 + 3e-5,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
    )
    builder.add_cylinder_visual(
        radius=radius * 1 / 5,
        half_length=thickness / 2 + 4e-5,
        material=sapien.render.RenderMaterial(base_color=TARGET_RED),
    )
    if add_collision:
        builder.add_cylinder_collision(
            radius=radius,
            half_length=thickness / 2,
        )
        builder.add_cylinder_collision(
            radius=radius * 4 / 5,
            half_length=thickness / 2 + 1e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 3 / 5,
            half_length=thickness / 2 + 2e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 2 / 5,
            half_length=thickness / 2 + 3e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 1 / 5,
            half_length=thickness / 2 + 4e-5,
        )
    return _build_by_type(builder, name, body_type)


def build_twocolor_peg(
    scene: ManiSkillScene,
    length,
    width,
    color_1,
    color_2,
    name: str,
    body_type="dynamic",
    add_collision: bool = True,
):
    builder = scene.create_actor_builder()
    if add_collision:
        builder.add_box_collision(
            half_size=[length, width, width],
        )
    builder.add_box_visual(
        pose=sapien.Pose(p=[-length / 2, 0, 0]),
        half_size=[length / 2, width, width],
        material=sapien.render.RenderMaterial(
            base_color=color_1,
        ),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[length / 2, 0, 0]),
        half_size=[length / 2, width, width],
        material=sapien.render.RenderMaterial(
            base_color=color_2,
        ),
    )
    return _build_by_type(builder, name, body_type)


RED_COLOR = [220 / 255, 12 / 255, 12 / 255, 1]
BLUE_COLOR = [0 / 255, 44 / 255, 193 / 255, 1]
GREEN_COLOR = [17 / 255, 190 / 255, 70 / 255, 1]


def build_fourcolor_peg(
    scene: ManiSkillScene,
    length,
    width,
    name: str,
    color_1=RED_COLOR,
    color_2=BLUE_COLOR,
    color_3=GREEN_COLOR,
    color_4=[1, 1, 1, 1],
    body_type="dynamic",
    add_collision: bool = True,
):
    """
    A peg with four sections and four different colors. Useful for visualizing every possible rotation without any symmetries
    """
    builder = scene.create_actor_builder()
    if add_collision:
        builder.add_box_collision(
            half_size=[length, width, width],
        )
    builder.add_box_visual(
        pose=sapien.Pose(p=[-length / 2, -width / 2, 0]),
        half_size=[length / 2, width / 2, width],
        material=sapien.render.RenderMaterial(
            base_color=color_1,
        ),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[length / 2, -width / 2, 0]),
        half_size=[length / 2, width / 2, width],
        material=sapien.render.RenderMaterial(
            base_color=color_2,
        ),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[-length / 2, width / 2, 0]),
        half_size=[length / 2, width / 2, width],
        material=sapien.render.RenderMaterial(
            base_color=color_3,
        ),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[length / 2, width / 2, 0]),
        half_size=[length / 2, width / 2, width],
        material=sapien.render.RenderMaterial(
            base_color=color_4,
        ),
    )
    return _build_by_type(builder, name, body_type)


def build_colorful_cube(
    scene: ManiSkillScene,
    half_size: float,
    color,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
):
    builder = scene.create_actor_builder()

    if add_collision:
        builder._mass = 0.1
        cube_material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=5, dynamic_friction=3, restitution=0
        )
        builder.add_box_collision(
            half_size=[half_size] * 3,
            material=cube_material,
        )
    builder.add_box_visual(
        half_size=[half_size] * 3,
        material=sapien.render.RenderMaterial(
            base_color=color,
        ),
    )
    return _build_by_type(builder, name, body_type)
