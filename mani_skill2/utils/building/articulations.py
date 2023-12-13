"""
Useful utilities for adding any object and geometry into a scene
"""

import os.path as osp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

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
    pass


@dataclass
class LinkMetadata:
    # semantic name of this link (e.g. handle)
    name: str
    link: physx.PhysxArticulationLinkComponent
    render_shapes: List[sapien.render.RenderShape]
    pass


@dataclass
class ArticulationMetadata:
    joints: Dict[str, JointMetadata]
    links: Dict[str, LinkMetadata]
    # a list of all movable links
    movable_links: List[str]


model_dbs: Dict[str, Dict[str, Dict]] = {}

# TODO optimization: we can cache some results in building articulations and reuse them


### Build articulations ###
def build_partnet_mobility_articulation(
    scene: sapien.Scene,
    model_id: str,
    fix_root_link=True,
    urdf_config: dict = None,
    set_object_on_ground=True,
):
    """
    Builds a physx.PhysxArticulation object into the scene and returns metadata containing annotations of the object's links and joints

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
    target_links = []
    target_joints = []
    target_handles = []

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

    # for link in self.articulation_metadata.movable_links:
    #         link_name = self.articulation_metadata.links[link].name
    #         b = self.articulation_metadata.joints[f"joint_{link.split('_')[1]}"].type
    #         c = self.articulation_metadata.joints[f"joint_{link.split('_')[1]}"].name
    #         print(link, link_name, b, c)

    return articulation, metadata


def _load_partnet_mobility_dataset():
    # load PartnetMobility
    model_dbs["PartnetMobility"] = {
        "model_data": load_json(
            PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_drawer_train.json"
        ),
        "builder": build_partnet_mobility_articulation,
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
