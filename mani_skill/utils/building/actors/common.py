"""
Common utilities for adding primitive prebuilt shapes to a scene
"""

from typing import Optional, Union

import numpy as np
import sapien
import sapien.render

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array


def _build_by_type(
    builder: ActorBuilder,
    name,
    body_type,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    if scene_idxs is not None:
        builder.set_scene_idxs(scene_idxs)
    if initial_pose is not None:
        builder.set_initial_pose(initial_pose)
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
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
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
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_box(
    scene: ManiSkillScene,
    half_sizes,
    color,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
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
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

def build_container_grid(
    scene: ManiSkillScene,
    size: float,
    height: float,
    thickness: float,
    color,
    name: str,
    n: int,
    m: int,
    initial_pose: sapien.Pose,
    body_type: str = "static",
    scene_idxs: Optional[Array] = None,
):
    builder = scene.create_actor_builder()

    # Container base
    base_pose = sapien.Pose([0., 0., -thickness / 2])  # Make the base's z equal to 0
    base_half_size = [size / 2, size / 2, thickness / 2]
    builder.add_box_collision(pose=base_pose, half_size=base_half_size)
    builder.add_box_visual(pose=base_pose, half_size=base_half_size)

    # Container sides (x4)
    for i in [-1, 1]:
        for axis in ['x', 'y']:
            side_pose = sapien.Pose(
                [i * (size - thickness) / 2 if axis == 'x' else 0,
                 i * (size - thickness) / 2 if axis == 'y' else 0,
                 height / 2]
            )
            side_half_size = [thickness / 2, size / 2, height / 2] if axis == 'x' else [size / 2, thickness / 2, height / 2]
            builder.add_box_collision(pose=side_pose, half_size=side_half_size)
            builder.add_box_visual(pose=side_pose, half_size=side_half_size)

    # Create grid cells
    internal_size = size - 2 * thickness

    cell_width = internal_size / n
    cell_height = internal_size / m

    for i in range(1, n):
        # Vertical dividers
        divider_pose = sapien.Pose([i * cell_width - internal_size / 2, 0, height / 2])
        divider_half_size = [thickness / 2, size / 2, height / 2]
        builder.add_box_collision(pose=divider_pose, half_size=divider_half_size)
        builder.add_box_visual(pose=divider_pose, half_size=divider_half_size)

    for j in range(1, m):
        # Horizontal dividers
        divider_pose = sapien.Pose([0, j * cell_height - internal_size / 2, height / 2])
        divider_half_size = [size / 2, thickness / 2, height / 2]
        builder.add_box_collision(pose=divider_pose, half_size=divider_half_size)
        builder.add_box_visual(pose=divider_pose, half_size=divider_half_size)

    container = _build_by_type(builder, name, body_type, scene_idxs, initial_pose)
    # Create goal sites at the center of each cell
    goal_sites = []
    goal_radius = 0.02
    goal_color = [0, 1, 0, 1]
    for i in range(n):
        for j in range(m):
            goal_x = (i + 0.5) * cell_width - internal_size / 2
            goal_y = (j + 0.5) * cell_height - internal_size / 2
            goal_local_pose = sapien.Pose([goal_x, goal_y, thickness + goal_radius])
            goal_world_pose = initial_pose * goal_local_pose  # Transform to world frame
            goal_site = build_box(
                scene,
                half_sizes=[cell_width / 2, cell_height / 2, thickness / 2],
                color=goal_color,
                name=f"goal_site_{i}_{j}",
                body_type="kinematic",
                add_collision=False,
                initial_pose=goal_world_pose,
            )
            goal_sites.append(goal_site)

    return container, goal_sites

def build_cylinder(
    scene: ManiSkillScene,
    radius: float,
    half_length: float,
    color,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    builder = scene.create_actor_builder()
    if add_collision:
        builder.add_cylinder_collision(
            radius=radius,
            half_length=half_length,
        )
    builder.add_cylinder_visual(
        radius=radius,
        half_length=half_length,
        material=sapien.render.RenderMaterial(
            base_color=color,
        ),
    )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_sphere(
    scene: ManiSkillScene,
    radius: float,
    color,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
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
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_red_white_target(
    scene: ManiSkillScene,
    radius: float,
    thickness: float,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
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
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_twocolor_peg(
    scene: ManiSkillScene,
    length,
    width,
    color_1,
    color_2,
    name: str,
    body_type="dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
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
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


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
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
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
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_colorful_cube(
    scene: ManiSkillScene,
    half_size: float,
    color,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
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
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)
