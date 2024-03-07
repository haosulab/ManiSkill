"""
Useful utilities for adding any object and geometry into a scene
"""

from pathlib import Path
from typing import Dict

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import transforms3d

from mani_skill import ASSET_DIR
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.io_utils import load_json

# map model dataset to a database of models
MODEL_DBS: Dict[str, Dict[str, Dict]] = {}


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


### Load individual assets ###


def build_actor(model_id: str, scene: ManiSkillScene, name: str):
    # TODO (stao): parse model id and determine from which dataset to pull asset from
    # e.g. YCB or our own sapien asset database in the future
    pass


### YCB Dataset ###


def build_actor_ycb(
    model_id: str,
    scene: ManiSkillScene,
    name: str,
    root_dir=ASSET_DIR / "assets/mani_skill2_ycb",
    body_type: str = "dynamic",
    add_collision: bool = True,
    return_builder: bool = False,
):
    if "YCB" not in MODEL_DBS:
        _load_ycb_dataset()
    model_db = MODEL_DBS["YCB"]["model_data"]  # TODO (stao): remove hardcode

    builder = scene.create_actor_builder()

    metadata = model_db[model_id]
    density = metadata.get("density", 1000)
    model_scales = metadata.get("scales", [1.0])
    scale = model_scales[0]
    physical_material = None
    height = (metadata["bbox"]["max"][2] - metadata["bbox"]["min"][2]) * scale
    model_dir = Path(root_dir) / "models" / model_id
    if add_collision:
        collision_file = str(model_dir / "collision.ply")
        builder.add_multiple_convex_collisions_from_file(
            filename=collision_file,
            scale=[scale] * 3,
            material=physical_material,
            density=density,
        )

    visual_file = str(model_dir / "textured.obj")
    builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)
    if return_builder:
        return builder, height
    return _build_by_type(builder, name, body_type), height


def _load_ycb_dataset():
    # load YCB if used
    MODEL_DBS["YCB"] = {
        "model_data": load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"),
        "builder": build_actor_ycb,
    }


### AI2THOR Object Dataset ###


def build_actor_ai2(
    model_id: str,
    scene: ManiSkillScene,
    name: str,
    kinematic: bool = False,
    set_object_on_ground=True,
):
    """
    Builds an actor/object from the AI2THOR assets.

    TODO (stao): Automatically makes the origin of the object be the center of the object.

    set_object_on_ground: bool
        if True, will set the pose of the created actor automatically so that the lowest point of the actor is at z = 0
    """
    model_path = (
        Path(ASSET_DIR)
        / "scene_datasets/ai2thor/ai2thorhab-uncompressed/assets/objects"
        / f"{model_id}.glb"
    )
    actor_id = name
    builder = scene.create_actor_builder()
    q = transforms3d.quaternions.axangle2quat(np.array([1, 0, 0]), theta=np.deg2rad(90))
    pose = sapien.Pose(q=q)
    builder.add_visual_from_file(str(model_path), pose=pose)
    if kinematic:
        builder.add_nonconvex_collision_from_file(str(model_path), pose=pose)
        actor = builder.build_kinematic(name=actor_id)
    else:
        builder.add_multiple_convex_collisions_from_file(
            str(model_path), decomposition="coacd", pose=pose
        )
        actor = builder.build(name=actor_id)

    if set_object_on_ground:
        actor.set_pose(sapien.Pose(p=[0, 0, 0]))
    return actor
