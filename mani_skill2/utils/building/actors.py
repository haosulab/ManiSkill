"""
Useful utilities for adding any object and geometry into a scene
"""

import os.path as osp
from pathlib import Path
from typing import Dict

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import transforms3d

from mani_skill2 import ASSET_DIR
from mani_skill2.utils.io_utils import load_json

# map model dataset to a database of models
model_dbs: Dict[str, Dict[str, Dict]] = {}


# Primitive Shapes
def build_cube(
    scene: sapien.Scene,
    half_size: float,
    color,
    name: str,
    dynamic: bool = True,
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
    if dynamic:
        actor = builder.build(name=name)
    else:
        actor = builder.build_static(name=name)
    return actor


def build_box(
    scene: sapien.Scene,
    half_sizes,
    color,
    name: str,
    dynamic: bool = True,
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
    if dynamic:
        actor = builder.build(name=name)
    else:
        actor = builder.build_static(name=name)
    return actor


def build_sphere(
    scene: sapien.Scene,
    radius: float,
    color,
    name: str,
    dynamic: bool = True,
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
    if dynamic:
        actor = builder.build(name=name)
    else:
        actor = builder.build_static(name=name)
    return actor


def build_twocolor_peg(
    scene: sapien.Scene,
    length,
    width,
    color_1,
    color_2,
    name: str,
    dynamic: bool = True,
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
    if dynamic:
        actor = builder.build(name=name)
    else:
        actor = builder.build_static(name=name)
    return actor


RED_COLOR = [220 / 255, 12 / 255, 12 / 255, 1]
BLUE_COLOR = [0 / 255, 44 / 255, 193 / 255, 1]
GREEN_COLOR = [17 / 255, 190 / 255, 70 / 255, 1]


def build_fourcolor_peg(
    scene: sapien.Scene,
    length,
    width,
    name: str,
    color_1=RED_COLOR,
    color_2=BLUE_COLOR,
    color_3=GREEN_COLOR,
    color_4=[1, 1, 1, 1],
    dynamic: bool = True,
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
    if dynamic:
        actor = builder.build(name=name)
    else:
        actor = builder.build_static(name=name)
    return actor


### Load individual assets ###


def build_actor(model_id: str, scene: sapien.Scene, name: str):
    # TODO (stao): parse model id and determine from which dataset to pull asset from
    # e.g. YCB or our own sapien asset database in the future
    pass


### YCB Dataset ###


def build_actor_ycb(
    model_id: str,
    scene: sapien.Scene,
    name: str,
    root_dir=ASSET_DIR / "mani_skill2_ycb",
):
    if "YCB" not in model_dbs:
        _load_ycb_dataset()
    model_db = model_dbs["YCB"]["model_data"]  # TODO (stao): remove hardcode

    builder = scene.create_actor_builder()

    density = model_db[model_id].get("density", 1000)
    model_scales = model_db[model_id].get("scales", [1.0])
    scale = model_scales[0]
    physical_material = None

    model_dir = Path(root_dir) / "models" / model_id
    collision_file = str(model_dir / "collision.obj")
    builder.add_multiple_convex_collisions_from_file(
        filename=collision_file,
        scale=[scale] * 3,
        material=physical_material,
        density=density,
        decomposition="coacd",
    )

    visual_file = str(model_dir / "textured.obj")
    builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)
    actor = builder.build(name=name)
    return actor


def _load_ycb_dataset():
    # load YCB if used
    model_dbs["YCB"] = {
        "model_data": load_json(ASSET_DIR / "mani_skill2_ycb/info_pick_v0.json"),
        "builder": build_actor_ycb,
    }


### AI2THOR Object Dataset ###


def build_actor_ai2(
    model_id: str,
    scene: sapien.Scene,
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

    aabb = actor.find_component_by_type(
        sapien.render.RenderBodyComponent
    ).compute_global_aabb_tight()
    height = aabb[1, 2] - aabb[0, 2]
    if set_object_on_ground:
        actor.set_pose(sapien.Pose(p=[0, 0, 0]))
    return actor
