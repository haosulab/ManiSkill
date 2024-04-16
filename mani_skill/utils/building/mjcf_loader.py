"""
Loader code to import MJCF formats.
Currently does not support loading motors in as this is not the standard SAPIEN adopts.
Instead Motor information can be fetched from the loader

Code partially adapted from https://github.com/NVIDIA/warp/blob/3ed2ceab824b65486c5204d2a7381d37b79fc314/warp/sim/import_mjcf.py
"""
import math
import os
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from typing import Any, Dict, List, Literal, Tuple, Union
from xml.etree.ElementTree import Element

import numpy as np

# from ..pysapien.physx import PhysxArticulation, PhysxMaterial
# from ..pysapien.render import RenderCameraComponent, RenderMaterial, RenderTexture2D
# from ..pysapien import Pose
import sapien
from sapien import ActorBuilder, Pose
from sapien.physx import PhysxArticulation, PhysxMaterial
from sapien.render import RenderCameraComponent, RenderMaterial, RenderTexture2D
from sapien.wrapper.articulation_builder import (
    ArticulationBuilder,
    LinkBuilder,
    MimicJointRecord,
)
from transforms3d import euler, quaternions


@dataclass
class MJCFTexture:
    name: str
    type: Literal["skybox", "cube", "2d"]
    rgb1: list
    rgb2: list


@dataclass
class MJCFDefault:
    parent: Element = None
    node: Element = None


def _parse_float(attrib, key, default):
    if key in attrib:
        return float(attrib[key])
    else:
        return default


def _str_to_float(string: str, delimiter=" "):
    res = [float(x) for x in string.split(delimiter)]
    if len(res) == 1:
        return res[0]
    return res


def _merge_attrib(default_attrib: dict, incoming_attribs: Union[List[dict], dict]):
    # from functools import reduce
    def helper_merge(a: dict, b: dict, path=[]):
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    helper_merge(a[key], b[key], path + [str(key)])
                else:
                    a[key] = b[key]
            else:
                a[key] = b[key]
        return a

    attrib = deepcopy(default_attrib)
    if isinstance(incoming_attribs, dict):
        incoming_attribs = [incoming_attribs]
    reduce(helper_merge, [attrib] + incoming_attribs)
    return attrib


def _parse_vec(attrib, key, default):
    if key in attrib:
        out = np.fromstring(attrib[key], sep=" ", dtype=np.float32)
    else:
        out = np.array(default, dtype=np.float32)
    return out


def _parse_orientation(attrib, use_degrees, euler_seq):
    if "quat" in attrib:
        wxyz = np.fromstring(attrib["quat"], sep=" ")
        return wxyz
    if "euler" in attrib:
        euler_angles = np.fromstring(attrib["euler"], sep=" ")
        if use_degrees:
            euler_angles *= np.pi / 180
        # TODO (stao): support other axes?
        return np.array(
            euler.euler2quat(euler_angles[0], euler_angles[1], euler_angles[2])
        )
    if "axisangle" in attrib:
        axisangle = np.fromstring(attrib["axisangle"], sep=" ")
        angle = axisangle[3]
        if use_degrees:
            angle *= np.pi / 180
        axis = axisangle[:3] / np.linalg.norm(axisangle[:3])
        return quaternions.axangle2quat(axis, angle)
    if "xyaxes" in attrib:
        xyaxes = np.fromstring(attrib["xyaxes"], sep=" ")
        # xaxis = wp.normalize(wp.vec3(*xyaxes[:3]))
        xaxis = xyaxes[:3] / np.linalg.norm(xyaxes[:3])
        # zaxis = wp.normalize(wp.vec3(*xyaxes[3:]))
        zaxis = xyaxes[3:] / np.linalg.norm(xyaxes[:3])
        # yaxis = wp.normalize(wp.cross(zaxis, xaxis))
        yaxis = np.cross(zaxis, xaxis)
        yaxis = yaxis / np.linalg.norm(yaxis)
        rot_matrix = np.array([xaxis, yaxis, zaxis]).T
        return quaternions.mat2quat(rot_matrix)
    if "zaxis" in attrib:
        zaxis = np.fromstring(attrib["zaxis"], sep=" ")
        # zaxis = wp.normalize(wp.vec3(*zaxis))
        zaxis = zaxis / np.linalg.norm(zaxis)
        # xaxis = wp.normalize(wp.cross(wp.vec3(0, 0, 1), zaxis))
        xaxis = np.cross(np.array([0, 0, 1]), zaxis)
        xaxis = xaxis / np.linalg.norm(xaxis)
        # yaxis = wp.normalize(wp.cross(zaxis, xaxis))
        yaxis = np.cross(zaxis, xaxis)
        yaxis = yaxis / np.linalg.norm(yaxis)
        rot_matrix = np.array([xaxis, yaxis, zaxis]).T
        return quaternions.mat2quat(rot_matrix)
    return np.array([1, 0, 0, 0])


class MJCFLoader:
    """
    Class to load MJCF into SAPIEN.
    """

    def __init__(self, ignore_classes=["motor"]):
        self.fix_root_link = True

        self.load_multiple_collisions_from_file = False
        self.multiple_collisions_decomposition = "none"
        self.multiple_collisions_decomposition_params = dict()

        self.collision_is_visual = False
        self.revolute_unwrapped = False
        self.scale = 1.0

        self.scene: sapien.Scene = None

        self.ignore_classes = ignore_classes

        # self._material = None
        # self._patch_radius = 0
        # self._min_patch_radius = 0
        self.density = 1000
        # self._link_material = dict()
        # self._link_patch_radius = dict()
        # self._link_min_patch_radius = dict()
        # self._link_density = dict()

        self._defaults: Dict[str, Element] = dict()
        self._assets = dict()
        self._materials = dict()
        self._textures: Dict[str, MJCFTexture] = dict()

        self._link2builder: Dict[str, LinkBuilder] = dict()
        self._link2parent_joint: Dict[str, Any] = dict()
        self._group_count = 0

    def set_scene(self, scene):
        self.scene = scene
        return self

    @staticmethod
    def _pose_from_origin(origin, scale):
        origin[:3, 3] = origin[:3, 3] * scale
        return Pose(origin)

    def _build_link(
        self, body: Element, body_attrib, link_builder: LinkBuilder, defaults
    ):
        """sets inertial, visual/collision shapes"""
        # inertial
        # TODO (stao)
        # if (
        #     link.inertial
        #     and link.inertial.mass != 0
        #     and not np.array_equal(link.inertial.inertia, np.zeros((3, 3)))
        # ):
        #     t_inertial2link = self._pose_from_origin(link.inertial.origin, self.scale)
        #     mass = link.inertial.mass
        #     inertia = link.inertial.inertia

        #     if np.array_equal(np.diag(np.diag(inertia)), inertia):
        #         eigs = np.diag(inertia)
        #         vecs = np.eye(3)
        #     else:
        #         eigs, vecs = np.linalg.eigh(inertia)
        #         if np.linalg.det(vecs) < 0:
        #             vecs[:, 2] = -vecs[:, 2]

        #     assert all([x > 0 for x in eigs]), "invalid moment of inertia"

        #     t_inertia2inertial = np.eye(4)
        #     t_inertia2inertial[:3, :3] = vecs
        #     t_inertia2inertial = Pose(t_inertia2inertial)

        #     t_inertial2link = t_inertial2link * t_inertia2inertial
        #     scale3 = self.scale**3
        #     scale5 = self.scale**5
        #     link_builder.set_mass_and_inertia(
        #         mass * scale3, t_inertial2link, scale5 * eigs
        #     )

        # go through each geometry of the body
        for geo_count, geom in enumerate(body.findall("geom")):
            geom_defaults = defaults
            if "class" in geom.attrib:
                geom_class = geom.attrib["class"]
                ignore_geom = False
                for pattern in self.ignore_classes:
                    if re.match(pattern, geom_class):
                        ignore_geom = True
                        break
                if ignore_geom:
                    continue
                if geom_class in self._defaults:
                    geom_defaults = _merge_attrib(defaults, self._defaults[geom_class])
            if "geom" in geom_defaults:
                geom_attrib = _merge_attrib(geom_defaults["geom"], geom.attrib)
            else:
                geom_attrib = geom.attrib

            geom_name = geom_attrib.get(
                "name", f"{body_attrib['name']}_geom_{geo_count}"
            )
            geom_type = geom_attrib.get("type", "sphere")
            if "mesh" in geom_attrib:
                geom_type = "mesh"

            geom_size = _parse_vec(geom_attrib, "size", [1.0, 1.0, 1.0]) * self.scale
            geom_pos = _parse_vec(geom_attrib, "pos", (0.0, 0.0, 0.0)) * self.scale
            geom_rot = _parse_orientation(
                geom_attrib, self._use_degrees, self._euler_seq
            )
            _parse_float(geom_attrib, "density", self.density)
            material = self._materials[geom_attrib["material"]]

            t_visual2link = Pose(geom_pos, geom_rot)
            if geom_type == "sphere":
                link_builder.add_sphere_visual(
                    t_visual2link, radius=geom_size[0], material=material
                )
            elif geom_type in ["capsule", "cylinder", "box"]:
                if "fromto" in geom_attrib:
                    geom_fromto = _parse_vec(
                        geom_attrib, "fromto", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                    )

                    start = np.array(geom_fromto[0:3]) * self.scale
                    end = np.array(geom_fromto[3:6]) * self.scale
                    # objects follow a line from start to end.

                    # objects are default along x-axis and we rotate accordingly via axis angle.
                    axis = (end - start) / np.linalg.norm(end - start)
                    # TODO this is bugged
                    angle = math.acos(np.dot(axis, np.array([1.0, 0.0, 0.0])))
                    axis = np.cross(axis, np.array([1.0, 0.0, 0.0]))
                    axis = axis / np.linalg.norm(axis)

                    geom_pos = (start + end) * 0.5
                    geom_rot = quaternions.axangle2quat(axis, -angle)
                    t_visual2link.set_p(geom_pos)
                    t_visual2link.set_q(geom_rot)
                    geom_radius = geom_size[0]
                    geom_half_length = np.linalg.norm(end - start) * 0.5
                else:
                    geom_radius = geom_size[0]
                    geom_half_length = geom_size[1]
                if geom_type == "capsule":
                    link_builder.add_capsule_visual(
                        t_visual2link,
                        radius=geom_radius,
                        half_length=geom_half_length,
                        material=material,
                        name=geom_name,
                    )
                    link_builder.add_capsule_collision(
                        t_visual2link,
                        radius=geom_radius,
                        half_length=geom_half_length,
                        # material=material,
                        # name=geom_name,
                    )
                else:
                    raise NotImplementedError()

            elif geom_type == "plane":
                pass
            elif geom_type == "ellipsoid":
                pass
            elif geom_type == "cylinder":
                pass
            elif geom_type == "mesh":
                pass
            elif geom_type == "sdf":
                raise NotImplementedError("SDF geom type not supported at the moment")
            elif geom_type == "hfield":
                raise NotImplementedError(
                    "Height fields are not supported at the moment"
                )

    def _parse_texture(self, texture: Element):
        """Parse MJCF textures to then be referenced by materials: https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-texture

        NOTE:
        - Procedural texture generation is currently not supported.
        - Different texture types are not really supported
        """
        name = texture.get("name")
        self._textures[name] = MJCFTexture(
            name=name,
            type=texture.get("type"),
            rgb1=texture.get("rgb1"),
            rgb2=texture.get("rgb2"),
        )

    def _parse_material(self, material: Element):
        """Parse MJCF materials in asset to sapien render materials"""
        name = material.get("name")
        # self._textures[material.get("texture")]
        # NOTE: Procedural texture generation is currently not supported.

        # Defaults from https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-material
        em_val = _str_to_float(material.get("emission", "0"))
        rgba = np.array(_str_to_float(material.get("rgba", "1 1 1 1")))
        self._materials[name] = RenderMaterial(
            emission=[rgba[0] * em_val, rgba[1] * em_val, rgba[2] * em_val, 1],
            base_color=rgba,
            specular=_str_to_float(material.get("specular", "0")),
            # TODO (stao): double check below 2 properties are right
            roughness=1 - _str_to_float(material.get("reflectance", "0")),
            metallic=_str_to_float(material.get("shininess", "0.5")),
        )

    @property
    def _root_default(self):
        return self._defaults["__root__"]

    def _parse_default(self, node: Element, parent: Element):
        """Parse a MJCF default attribute. https://mujoco.readthedocs.io/en/stable/modeling.html#default-settings explains how it works"""
        class_name = "__root__"
        if node.tag == "default":
            if "class" in node.attrib:
                class_name = node.attrib["class"]
            if parent is not None and "class" in parent.attrib:
                self._defaults[class_name] = deepcopy(
                    self._defaults[parent.attrib["class"]]
                )
            else:
                self._defaults[class_name] = {}
        for child in node:
            if child.tag == "default":
                self._parse_default(child, node)
            else:
                if child.tag in self._defaults[class_name]:
                    self._defaults[class_name][child.tag] = _merge_attrib(
                        self._defaults[class_name][child.tag], child.attrib
                    )
                else:
                    self._defaults[class_name][child.tag] = child.attrib

    def _parse_body(
        self,
        body: Element,
        parent: LinkBuilder,
        incoming_defaults: dict,
        builder: ArticulationBuilder,
    ):
        body_class = body.get("childclass")
        if body_class is None:
            defaults = incoming_defaults
        else:
            for pattern in self.ignore_classes:
                if re.match(pattern, body_class):
                    return
            defaults = _merge_attrib(incoming_defaults, self._defaults[body_class])

        if "body" in defaults:
            body_attrib = _merge_attrib(defaults["body"], body.attrib)
        else:
            body_attrib = body.attrib

        body_name = body_attrib["name"]
        body_pos = _parse_vec(body_attrib, "pos", (0.0, 0.0, 0.0))
        body_ori = _parse_orientation(
            body_attrib, use_degrees=self._use_degrees, euler_seq=self._euler_seq
        )
        # if parent is None:
        #     # transform which way is up
        #     body_pos = wp.transform_point(xform, body_pos)
        #     body_ori = xform.q * body_ori
        body_pos *= self.scale
        body_pose = Pose(body_pos)

        # TODO (stao): Support free joints? Are they just fix_root_link=False perhaps?
        freejoint_tags = body.findall("freejoint")
        if False and len(freejoint_tags) > 0:
            link_builder = builder.create_link_builder(parent=parent)
            link_builder.set_name(f"{body_name}")
            self._build_link(body, body_attrib, link_builder, defaults)
        else:
            link_builder = parent
            # if body has no joints, it is a fixed joint
            joints = body.findall("joint")
            if len(joints) == 0:
                joints = [ET.Element("joint", attrib=dict(type="fixed"))]
            for i, joint in enumerate(joints):
                # note there can be multiple joints here. We create some dummy links to simulate that

                # order of defaults is current inherited defaults, then class, the joint.attrib
                incoming_attributes = []
                if "joint" in defaults:
                    incoming_attributes.append(defaults["joint"])
                if "class" in joint.attrib:
                    incoming_attributes.append(
                        self._defaults[joint.attrib["class"]]["joint"]
                    )
                incoming_attributes.append(joint.attrib)
                joint_attrib = _merge_attrib(dict(), incoming_attributes)

                # create a dummy link
                if i >= 0:
                    link_builder = builder.create_link_builder(parent=link_builder)
                    link_builder.set_name(f"{body_name}_{i}")
                    self._link2builder[link_builder.name] = link_builder
                link_builder.set_joint_name(joint_attrib.get("name", ""))
                joint_type = joint_attrib.get("type", "hinge")
                np.array(_parse_vec(joint_attrib, "pos", [0, 0, 0]))
                if i == len(joints) - 1:
                    # the last link is the "real" one, the rest are dummy links to support multiple joints acting on a link
                    self._build_link(body, body_attrib, link_builder, defaults)
                """
                Notes:

                Joint Stifness is controlled via controller stiffness values
                """
                # import ipdb;ipdb.set_trace()
                t_joint2parent = Pose()  # body_pose  # * Pose(-joint_origin)
                if i == 0:
                    t_joint2parent = body_pose

                friction = 0
                damping = 0
                # TODO (stao): fix joint dynamics
                # if joint.dynamics:
                #     friction = joint.dynamics.friction
                #     damping = joint.dynamics.damping

                axis = _parse_vec(joint_attrib, "axis", [0.0, 0.0, 0.0])
                axis_norm = np.linalg.norm(axis)
                if axis_norm < 1e-3:
                    axis = np.array([1.0, 0.0, 0.0])
                else:
                    axis /= axis_norm

                if abs(axis @ [1.0, 0.0, 0.0]) > 0.9:
                    axis1 = np.cross(axis, [0.0, 0.0, 1.0])
                    axis1 /= np.linalg.norm(axis1)
                else:
                    axis1 = np.cross(axis, [1.0, 0.0, 0.0])
                    axis1 /= np.linalg.norm(axis1)
                axis2 = np.cross(axis, axis1)
                t_axis2joint = np.eye(4)
                t_axis2joint[:3, 0] = axis
                t_axis2joint[:3, 1] = axis1
                t_axis2joint[:3, 2] = axis2
                t_axis2joint = Pose(t_axis2joint)
                t_axis2parent = t_joint2parent * t_axis2joint
                # if "range" not in joint_attrib:
                #     assert (
                #         joint_type == "fixed"
                #     ) and "limited" not in joint_attrib or not joint_attrib["limited"], "Found a joint that is not fixed, limited is not False, and has no range"
                if joint_type == "hinge":
                    limited = joint_attrib.get("limited", "true")
                    if limited == "true":
                        joint_limits = _parse_vec(joint_attrib, "range", [0, 0])
                        if self._use_degrees:
                            joint_limits = np.deg2rad(joint_limits)
                        link_builder.set_joint_properties(
                            "revolute_unwrapped",
                            [joint_limits],
                            t_axis2parent,
                            t_axis2joint,
                            friction,
                            damping,
                        )
                    else:
                        link_builder.set_joint_properties(
                            "revolute",
                            [[-np.inf, np.inf]],
                            t_axis2parent,
                            t_axis2joint,
                            friction,
                            damping,
                        )
                elif joint_type == "slide":
                    link_builder.set_joint_properties(
                        "prismatic",
                        [_parse_vec(joint_attrib, "range", [0, 0]) * self.scale],
                        t_axis2parent,
                        t_axis2joint,
                        friction,
                        damping,
                    )
                elif joint_type == "fixed":
                    # TODO (stao): how does mjcf do fixed joints?
                    link_builder.set_joint_properties(
                        "fixed",
                        [],
                        t_axis2parent,
                        t_axis2joint,
                        friction,
                        damping,
                    )

            # ensure adjacent links do not collide
            if parent is not None:
                parent.collision_groups[2] |= 1 << (self._group_count)
                link_builder.collision_groups[2] |= 1 << (self._group_count)
                self._group_count += 1

        for child in body.findall("body"):
            self._parse_body(child, link_builder, defaults, builder)

    def _parse_constraint(self, constraint: Element):
        joint_elems = []
        for joint in constraint.findall("joint"):
            joint_elems.append(joint)
        return MimicJointRecord(
            joint_elems[0].attrib["joint"],
            joint_elems[1].attrib["joint"],
            1,
            0
            # joint.mimic.multiplier,
            # joint.mimic.offset,
        )

    def _parse_mjcf(
        self, mjcf_string: str
    ) -> Tuple[List[ArticulationBuilder], List[ActorBuilder], None]:
        """Helper function for self.parse"""
        xml: Element = ET.fromstring(mjcf_string.encode("utf-8"))
        # import ipdb;ipdb.set_trace()
        # robot = URDF._from_xml(xml, self.urdf_dir, lazy_load_meshes=True)

        self._use_degrees = True  # angles are in degrees by default
        self._euler_seq = [1, 2, 3]  # XYZ by default

        compiler = xml.find("compiler")
        if compiler is not None:
            self._use_degrees = (
                compiler.attrib.get("angle", "degree").lower() == "degree"
            )
            self._euler_seq = [
                "xyz".index(c) + 1
                for c in compiler.attrib.get("eulerseq", "xyz").lower()
            ]
            self._mesh_dir = compiler.attrib.get("meshdir", ".")

        ### Parse assets ###
        for asset in xml.findall("asset"):
            for texture in asset.findall("texture"):
                self._parse_texture(texture)
            for material in asset.findall("material"):
                self._parse_material(material)
        ### Parse defaults ###
        for default in xml.findall("default"):
            self._parse_default(default, None)

        ### Parse World Body ###

        # NOTE (stao): For now we assume there is only one articulation. Some setups like Aloha 2 are technically 2 articulations
        # but you can treat it as a single one anyway
        builder = self.scene.create_articulation_builder()
        for body in xml.find("worldbody").findall("body"):
            dummy_root_link = builder.create_link_builder(None)
            dummy_root_link.name = "dummy_root"
            dummy_root_link.set_joint_properties(
                type="fixed", limits=None, pose_in_parent=Pose(), pose_in_child=Pose()
            )
            self._parse_body(body, dummy_root_link, self._root_default, builder)

        ### Parse contact and exclusions ###
        for contact in xml.findall("contact"):
            # self._parse_contact()
            pass

        ### Parse equality constraints ###
        tendon = xml.find("tendon")
        if tendon is not None:
            for constraint in tendon.findall("fixed"):
                record = self._parse_constraint(constraint)
                builder.mimic_joint_records.append(record)

        return [builder], [], []

    def parse(self, mjcf_file: str, package_dir=None):
        """Parses a given MJCF file into articulation builders, actor builders, and sensors"""
        self.package_dir = package_dir
        self.mjcf_dir = os.path.dirname(mjcf_file)

        with open(mjcf_file, "r") as f:
            mjcf_string = f.read()

        return self._parse_mjcf(mjcf_string)

    def load(self, mjcf_file: str, package_dir=None):
        articulation_builders, actor_builders, cameras = self.parse(
            mjcf_file, package_dir
        )

        articulations: List[PhysxArticulation] = []
        for b in articulation_builders:
            articulations.append(b.build())

        actors = []
        for b in actor_builders:
            actors.append(b.build())

        # TODO (stao): how does mjcf specify sensors?
        # name2entity = dict()
        # for a in articulations:
        #     for l in a.links:
        #         name2entity[l.name] = l.entity

        # for a in actors:
        #     name2entity[a.name] = a
        return articulations[0]
