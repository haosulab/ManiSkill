"""
Useful utilities for adding any object and geometry into a scene
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
from sapien import Pose

from mani_skill2 import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill2.utils.geometry.trimesh_utils import (
    get_articulation_meshes,
    merge_meshes,
)
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.sapien_utils import apply_urdf_config, parse_urdf_config


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
model_dbs: Dict[str, Dict[str, Dict]] = {}


### Build articulations ###
def build_preprocessed_partnet_mobility_articulation(
    scene: sapien.Scene,
    model_id: str,
    fix_root_link=True,
    urdf_config: dict = None,
    set_object_on_ground=True,
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
    if "PartnetMobility" not in model_dbs:
        _load_partnet_mobility_dataset()

    metadata = model_dbs["PartnetMobility"]["model_data"][model_id]

    loader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    loader.scale = metadata["scale"]
    loader.load_multiple_collisions_from_file = True

    urdf_path = model_dbs["PartnetMobility"]["model_urdf_paths"][model_id]
    urdf_config = parse_urdf_config(urdf_config or {}, scene)
    apply_urdf_config(loader, urdf_config)
    articulation: physx.PhysxArticulation = loader.load(str(urdf_path))

    metadata = ArticulationMetadata(joints=dict(), links=dict(), movable_links=[])

    # NOTE(jigu): links and their parent joints.
    for link, joint in zip(articulation.get_links(), articulation.get_joints()):
        metadata.joints[joint.name] = JointMetadata(type=joint.type, name="")
        render_body = link.entity.find_component_by_type(
            sapien.render.RenderBodyComponent
        )
        render_shapes = []
        if render_body is not None:
            render_shapes = render_body.render_shapes
        metadata.links[link.name] = LinkMetadata(
            name=None, link=link, render_shapes=render_shapes
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

    if set_object_on_ground:
        qlimits = articulation.get_qlimits()  # [N, 2]
        assert not np.isinf(qlimits).any(), qlimits
        qpos = np.ascontiguousarray(qlimits[:, 0])
        articulation.set_qpos(qpos)
        articulation.set_pose(Pose())
        bounds = merge_meshes(get_articulation_meshes(articulation)).bounds
        articulation.set_pose(Pose([0, 0, -bounds[0, 2]]))

    return articulation, metadata


def _load_partnet_mobility_dataset():
    """loads preprocssed partnet mobility metadata"""
    model_dbs["PartnetMobility"] = {
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

    model_dbs["PartnetMobility"]["model_urdf_paths"] = {
        k: find_urdf_path(k) for k in model_dbs["PartnetMobility"]["model_data"].keys()
    }
