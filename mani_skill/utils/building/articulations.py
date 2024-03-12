"""
Useful utilities for adding any object and geometry into a scene
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import trimesh
from sapien import Pose

from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.trimesh_utils import (
    get_articulation_meshes,
    merge_meshes,
)
from mani_skill.utils.io_utils import load_json


@dataclass
class JointMetadata:
    type: str
    # semantic name of this joint
    name: str


@dataclass
class LinkMetadata:
    # semantic name of this link (e.g. handle)
    name: str
    link: physx.PhysxArticulationLinkComponent
    render_shapes: List[sapien.render.RenderShape]


@dataclass
class ArticulationMetadata:
    joints: Dict[str, JointMetadata]
    links: Dict[str, LinkMetadata]
    # a list of all movable links
    movable_links: List[str]
    bbox: trimesh.primitives.Box
    scale: float


def build_articulation_from_file(
    scene: sapien.Scene,
    urdf_path: str,
    fix_root_link=True,
    scale: float = 1.0,
    decomposition="none",
    set_object_on_ground=True,
):
    loader = scene.create_urdf_loader()
    loader.multiple_collisions_decomposition = decomposition
    loader.fix_root_link = fix_root_link
    loader.scale = scale
    loader.load_multiple_collisions_from_file = True
    articulation: physx.PhysxArticulation = loader.load(urdf_path)
    articulation.set_qpos(articulation.qpos)
    bounds = merge_meshes(get_articulation_meshes(articulation)).bounds
    if set_object_on_ground:
        articulation.set_pose(Pose([0, 0, -bounds[0, 2]]))
    return articulation, bounds


# cache model metadata here if needed
MODEL_DBS: Dict[str, Dict[str, Dict]] = {}


### Build articulations ###
def build_preprocessed_partnet_mobility_articulation(
    scene: ManiSkillScene,
    model_id: str,
    name: str,
    fix_root_link=True,
    urdf_config: dict = None,
    scene_idxs=None,
):
    """
    Builds a physx.PhysxArticulation object into the scene and returns metadata containing annotations of the object's links and joints.

    This uses preprocessed data from the ManiSkill team where assets were annotated with correct scales and provided
    proper convex decompositions of the articulations.

    Args:
        scene: the sapien scene to add articulation to
        model_id: the id of the partnet mobility model to load

        set_object_on_ground: whether to change the pose of the built articulation such that the object is settled on the ground (at z = 0)
    """
    if "PartnetMobility" not in MODEL_DBS:
        _load_partnet_mobility_dataset()

    metadata = MODEL_DBS["PartnetMobility"]["model_data"][model_id]

    loader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    loader.scale = metadata["scale"]
    loader.load_multiple_collisions_from_file = True
    # loader.multiple_collisions_decomposition="coacd"
    loader.disable_self_collisions = True
    urdf_path = MODEL_DBS["PartnetMobility"]["model_urdf_paths"][model_id]
    urdf_config = sapien_utils.parse_urdf_config(urdf_config or {}, scene)
    sapien_utils.apply_urdf_config(loader, urdf_config)
    articulation = loader.load(str(urdf_path), name=name, scene_idxs=scene_idxs)
    metadata = ArticulationMetadata(
        joints=dict(), links=dict(), movable_links=[], bbox=None, scale=loader.scale
    )

    for link, joint in zip(articulation.get_links(), articulation.get_joints()):
        metadata.joints[joint.name] = JointMetadata(type=joint.type, name="")
        # render_body = link.entity.find_component_by_type(
        #     sapien.render.RenderBodyComponent
        # )
        # render_shapes = []
        # if render_body is not None:
        #     render_shapes = render_body.render_shapes
        metadata.links[link.name] = LinkMetadata(
            name=None,
            link=link,
            render_shapes=[],  # render_shapes=render_shapes
        )
        if joint.type != "fixed":
            metadata.movable_links.append(link.name)

    # parse semantic information about each link and the joint controlling it
    with open(
        ASSET_DIR / "partnet_mobility/dataset" / str(model_id) / "semantics.txt", "r"
    ) as f:
        for line in f.readlines():
            link_id, joint_type, link_name = line[:-1].split(" ")
            metadata.links[link_id].name = link_name
            metadata.joints[f"joint_{link_id.split('_')[1]}"].name = joint_type

    bbox = merge_meshes(get_articulation_meshes(articulation._objs[0])).bounding_box
    metadata.bbox = bbox
    return articulation, metadata


def _load_partnet_mobility_dataset():
    """loads preprocssed partnet mobility metadata"""
    MODEL_DBS["PartnetMobility"] = {
        "model_data": load_json(
            PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_drawer_train.json"
        ),
    }

    def find_urdf_path(model_id):
        model_dir = ASSET_DIR / "partnet_mobility/dataset" / str(model_id)
        urdf_names = ["mobility_cvx.urdf", "mobility_fixed.urdf"]
        for urdf_name in urdf_names:
            urdf_path = model_dir / urdf_name
            if urdf_path.exists():
                return urdf_path

    MODEL_DBS["PartnetMobility"]["model_urdf_paths"] = {
        k: find_urdf_path(k) for k in MODEL_DBS["PartnetMobility"]["model_data"].keys()
    }


def build_robel_valve(
    scene: ManiSkillScene,
    valve_angles: Sequence[float],
    name: str,
    radius_scale: float = 1.0,
    capsule_radius_scale: float = 1.0,
    scene_idxs=None,
):
    # Size and geometry of valve are based on the original setting of Robel benchmark, unit: m
    # Ref: https://github.com/google-research/robel
    capsule_height = 0.039854
    capsule_length = 0.061706 * radius_scale
    capsule_radius = 0.0195 * capsule_radius_scale
    bottom_length = 0.04
    bottom_height = 0.03
    bearing_radius = 0.007
    bearing_height = 0.032

    builder = scene.create_articulation_builder()
    builder.set_scene_idxs(scene_idxs)

    # Mount link
    mount_builder = builder.create_link_builder(parent=None)
    mount_builder.set_name("mount")
    mount_builder.add_box_collision(
        pose=sapien.Pose([0, 0, bottom_height / 2]),
        half_size=[bottom_length / 2, bottom_length / 2, bottom_height / 2],
    )
    mount_builder.add_box_visual(
        pose=sapien.Pose([0, 0, bottom_height / 2]),
        half_size=[bottom_length / 2, bottom_length / 2, bottom_height / 2],
    )
    mount_builder.add_cylinder_visual(
        pose=sapien.Pose(
            [0, 0, bottom_height + bearing_height / 2], [-0.707, 0, 0.707, 0]
        ),
        half_length=bottom_height / 2,
        radius=bearing_radius,
    )
    mount_builder.add_cylinder_collision(
        pose=sapien.Pose(
            [0, 0, bottom_height + bearing_height / 2], [-0.707, 0, 0.707, 0]
        ),
        half_length=bottom_height / 2,
        radius=bearing_radius,
    )

    # Valve link
    valve_builder = builder.create_link_builder(mount_builder)
    valve_builder.set_name("valve")
    valve_angles = np.array(valve_angles)
    if np.min(valve_angles) < 0 or np.max(valve_angles) > 2 * np.pi:
        raise ValueError(
            f"valve_angles should be within 0-2*pi, but got {valve_angles}"
        )

    for i, angle in enumerate(valve_angles):
        rotate_pose = sapien.Pose([0, 0, 0])
        rotate_pose.set_rpy([0, 0, angle])
        capsule_pose = rotate_pose * sapien.Pose([capsule_length / 2, 0, 0])
        color = np.array([1, 1, 1, 1]) if i > 0 else np.array([1, 0, 0, 1])
        viz_mat = sapien.render.RenderMaterial(
            base_color=color, roughness=0.5, specular=0.5
        )
        valve_builder.add_capsule_visual(
            pose=capsule_pose,
            radius=capsule_radius,
            half_length=capsule_length / 2,
            material=viz_mat,
        )
        physx_mat = sapien.physx.PhysxMaterial(1, 0.8, 0)
        valve_builder.add_capsule_collision(
            pose=capsule_pose,
            radius=capsule_radius,
            half_length=capsule_length / 2,
            material=physx_mat,
            patch_radius=0.1,
            min_patch_radius=0.03,
        )

    valve_builder.set_joint_name("valve_joint")
    valve_builder.set_joint_properties(
        type="revolute",
        limits=[[-np.inf, np.inf]],
        pose_in_parent=sapien.Pose(
            [0, 0, capsule_height + bottom_height], [0.707, 0, 0.707, 0]
        ),
        pose_in_child=sapien.Pose(q=[0.707, 0, 0.707, 0]),
        friction=0.02,
        damping=2,
    )

    valve = builder.build(name, fix_root_link=True)
    return valve, capsule_length
