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
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple
from xml.etree.ElementTree import Element

import numpy as np

# from ..pysapien.physx import PhysxArticulation, PhysxMaterial
# from ..pysapien.render import RenderCameraComponent, RenderMaterial, RenderTexture2D
# from ..pysapien import Pose
import sapien
from sapien import ActorBuilder, Pose
from sapien.physx import PhysxArticulation, PhysxMaterial
from sapien.render import RenderCameraComponent, RenderMaterial, RenderTexture2D
from sapien.wrapper.articulation_builder import ArticulationBuilder, LinkBuilder
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


def _merge_attrib(default_attrib: dict, incoming_attrib: dict):
    attrib = default_attrib.copy()
    attrib.update(incoming_attrib)
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
        zaxis = wp.normalize(wp.vec3(*zaxis))
        xaxis = wp.normalize(wp.cross(wp.vec3(0, 0, 1), zaxis))
        yaxis = wp.normalize(wp.cross(zaxis, xaxis))
        rot_matrix = np.array([xaxis, yaxis, zaxis]).T
        return wp.quat_from_matrix(rot_matrix)
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

        self._link2builder: Dict[str, LinkBuilder]
        self._link2parent_joint: Dict[str, Any]

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
            # t_visual2link = self._pose_from_origin(visual.origin, self.scale)
            t_visual2link = Pose(geom_pos, geom_rot)
            if geom_type == "sphere":
                pass
            elif geom_type == "box":
                pass
            elif geom_type in ["capsule", "cylinder"]:
                if "fromto" in geom_attrib:
                    geom_fromto = parse_vec(
                        geom_attrib, "fromto", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                    )

                    start = wp.vec3(geom_fromto[0:3]) * scale
                    end = wp.vec3(geom_fromto[3:6]) * scale

                    # compute rotation to align the Warp capsule (along x-axis), with mjcf fromto direction
                    axis = wp.normalize(end - start)
                    angle = math.acos(wp.dot(axis, wp.vec3(0.0, 1.0, 0.0)))
                    axis = wp.normalize(wp.cross(axis, wp.vec3(0.0, 1.0, 0.0)))

                    geom_pos = (start + end) * 0.5
                    geom_rot = wp.quat_from_axis_angle(axis, -angle)

                    geom_size[0]
                    wp.length(end - start) * 0.5
                link_builder.add_capsule_visual(
                    t_visual2link * Pose(q=[0.7071068, 0, 0.7071068, 0]),
                    visual.geometry.capsule.radius * self.scale,
                    visual.geometry.capsule.length * self.scale / 2.0,
                    material=material,
                    name=name,
                )
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

            import ipdb

            ipdb.set_trace()

        # visual shapes
        for visual in link.visuals:
            material = None
            if visual.material:
                material = RenderMaterial()
                if visual.material.color is not None:
                    material.base_color = visual.material.color
                elif visual.material.texture is not None:
                    material.diffuse_texture = RenderTexture2D(
                        _try_very_hard_to_find_file(
                            visual.material.texture.filename,
                            self.urdf_dir,
                            self.package_dir,
                        )
                    )

            t_visual2link = self._pose_from_origin(visual.origin, self.scale)
            name = visual.name if visual.name else ""
            if visual.geometry.box:
                link_builder.add_box_visual(
                    t_visual2link,
                    visual.geometry.box.size * self.scale / 2.0,
                    material=material,
                    name=name,
                )
            if visual.geometry.sphere:
                link_builder.add_sphere_visual(
                    t_visual2link,
                    visual.geometry.sphere.radius * self.scale,
                    material=material,
                    name=name,
                )
            if visual.geometry.capsule:
                link_builder.add_capsule_visual(
                    t_visual2link * Pose(q=[0.7071068, 0, 0.7071068, 0]),
                    visual.geometry.capsule.radius * self.scale,
                    visual.geometry.capsule.length * self.scale / 2.0,
                    material=material,
                    name=name,
                )
            if visual.geometry.cylinder:
                link_builder.add_cylinder_visual(
                    t_visual2link * Pose(q=[0.7071068, 0, 0.7071068, 0]),
                    visual.geometry.cylinder.radius * self.scale,
                    visual.geometry.cylinder.length * self.scale / 2.0,
                    material=material,
                    name=name,
                )
            if visual.geometry.mesh:
                if visual.geometry.mesh.scale is not None:
                    scale = visual.geometry.mesh.scale
                else:
                    scale = np.ones(3)

                link_builder.add_visual_from_file(
                    _try_very_hard_to_find_file(
                        visual.geometry.mesh.filename,
                        self.urdf_dir,
                        self.package_dir,
                    ),
                    t_visual2link,
                    scale * self.scale,
                    material=material,
                    name=name,
                )

        # collision shapes
        for cid, collision in enumerate(link.collisions):
            t_collision2link = self._pose_from_origin(collision.origin, self.scale)

            material = self._get_material(link.name, cid)
            density = self._get_density(link.name, cid)
            patch_radius = self._get_patch_radius(link.name, cid)
            min_patch_radius = self._get_min_patch_radius(link.name, cid)

            if collision.geometry.box:
                link_builder.add_box_collision(
                    t_collision2link,
                    collision.geometry.box.size * self.scale / 2.0,
                    material=material,
                    density=density,
                    patch_radius=patch_radius,
                    min_patch_radius=min_patch_radius,
                )
                if self.collision_is_visual:
                    link_builder.add_box_visual(
                        t_collision2link,
                        collision.geometry.box.size * self.scale / 2.0,
                    )
            if collision.geometry.sphere:
                link_builder.add_sphere_collision(
                    t_collision2link,
                    collision.geometry.sphere.radius * self.scale,
                    material=material,
                    density=density,
                    patch_radius=patch_radius,
                    min_patch_radius=min_patch_radius,
                )
                if self.collision_is_visual:
                    link_builder.add_sphere_visual(
                        t_collision2link,
                        collision.geometry.sphere.radius * self.scale,
                    )
            if collision.geometry.capsule:
                link_builder.add_capsule_collision(
                    t_collision2link * Pose(q=[0.7071068, 0, 0.7071068, 0]),
                    collision.geometry.capsule.radius * self.scale,
                    collision.geometry.capsule.length * self.scale / 2.0,
                    material=material,
                    density=density,
                    patch_radius=patch_radius,
                    min_patch_radius=min_patch_radius,
                )
                if self.collision_is_visual:
                    link_builder.add_capsule_visual(
                        t_collision2link * Pose(q=[0.7071068, 0, 0.7071068, 0]),
                        collision.geometry.capsule.radius * self.scale,
                        collision.geometry.capsule.length * self.scale / 2.0,
                    )
            if collision.geometry.cylinder:
                link_builder.add_cylinder_collision(
                    t_collision2link * Pose(q=[0.7071068, 0, 0.7071068, 0]),
                    collision.geometry.cylinder.radius * self.scale,
                    collision.geometry.cylinder.length * self.scale / 2.0,
                    material=material,
                    density=density,
                    patch_radius=patch_radius,
                    min_patch_radius=min_patch_radius,
                )
                if self.collision_is_visual:
                    link_builder.add_cylinder_visual(
                        t_collision2link * Pose(q=[0.7071068, 0, 0.7071068, 0]),
                        collision.geometry.cylinder.radius * self.scale,
                        collision.geometry.cylinder.length * self.scale / 2.0,
                    )

            if collision.geometry.mesh:
                if collision.geometry.mesh.scale is not None:
                    scale = collision.geometry.mesh.scale
                else:
                    scale = np.ones(3)

                filename = _try_very_hard_to_find_file(
                    collision.geometry.mesh.filename,
                    self.urdf_dir,
                    self.package_dir,
                )

                if self.load_multiple_collisions_from_file:
                    link_builder.add_multiple_convex_collisions_from_file(
                        filename,
                        t_collision2link,
                        scale * self.scale,
                        material=material,
                        density=density,
                        patch_radius=patch_radius,
                        min_patch_radius=min_patch_radius,
                        decomposition=self.multiple_collisions_decomposition,
                        decomposition_params=self.multiple_collisions_decomposition_params,
                    )
                else:
                    link_builder.add_convex_collision_from_file(
                        filename,
                        t_collision2link,
                        scale * self.scale,
                        material=material,
                        density=density,
                        patch_radius=patch_radius,
                        min_patch_radius=min_patch_radius,
                    )

                if self.collision_is_visual:
                    link_builder.add_visual_from_file(
                        filename,
                        t_collision2link,
                        scale * self.scale,
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
        self._textures[material.get("texture")]
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
            self._defaults[class_name] = {}
        for child in node:
            if child.tag == "default":
                self._parse_default(child, node)
            else:
                self._defaults[class_name][child.tag] = child.attrib

    def _parse_body(
        self,
        body: Element,
        parent: Element,
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

        joint_type = None

        link_builder = builder.create_link_builder(parent=parent)
        link_builder.set_name(f"{body_name}")
        # TODO (stao): Support free joints? Are they just fix_root_link=False perhaps?
        freejoint_tags = body.findall("freejoint")
        if len(freejoint_tags) > 0:
            pass
        else:
            joints = body.findall("joint")
            for i, joint in enumerate(joints):
                # note there can be multiple joints here. We create some dummy links to simulate that
                if "joint" in defaults:
                    joint_attrib = _merge_attrib(defaults["joint"], joint.attrib)
                else:
                    joint_attrib = joint.attrib

                # create a dummy link
                if i > 0:
                    link_builder = builder.create_link_builder(parent=parent)
                    link_builder.set_name(f"{body_name}_{i}")
                    self._link2builder[link_builder.name] = link_builder
                link_builder.set_joint_name(
                    joint_attrib.get("name", "")
                    # self.link2parent_joint[link_name].name
                    # if self.link2parent_joint[link_name] is not None
                    # else ""
                )

                if i == 0:
                    # first link is the real one, the rest are dummy links to support multiple joints acting on a link
                    self._build_link(body, body_attrib, link_builder, defaults)

                joint_type = joint_attrib.get("type", "hinge")
                joint_origin = np.array(_parse_vec(joint_attrib, "pos", [0, 0, 0]))
                t_joint2parent = (
                    self._pose_from_origin(joint_origin, self.scale)
                    if joint
                    else Pose()
                )

                friction = 0
                damping = 0
                # TODO (stao): fix joint dynamics
                # if joint.dynamics:
                #     friction = joint.dynamics.friction
                #     damping = joint.dynamics.damping

                axis = _parse_vec(joint_attrib, "axis", [0.0, 0.0, 0.0])
                axis_norm = np.linalg.norm(axis)
                if axis_norm < 1e-3:
                    axis = np.array([1, 0, 0])
                else:
                    axis /= axis_norm

                if abs(axis @ [1, 0, 0]) > 0.9:
                    axis1 = np.cross(axis, [0, 0, 1])
                    axis1 /= np.linalg.norm(axis1)
                else:
                    axis1 = np.cross(axis, [1, 0, 0])
                    axis1 /= np.linalg.norm(axis1)
                axis2 = np.cross(axis, axis1)
                t_axis2joint = np.eye(4)
                t_axis2joint[:3, 0] = axis
                t_axis2joint[:3, 1] = axis1
                t_axis2joint[:3, 2] = axis2
                t_axis2joint = Pose(t_axis2joint)
                t_axis2parent = t_joint2parent * t_axis2joint

                if joint_type == "hinge":
                    link_builder.set_joint_properties(
                        "revolute",
                        [_parse_vec(joint_attrib, "range", [0, 0])],
                        # [[joint.limit.lower, joint.limit.upper]],
                        t_axis2parent,
                        t_axis2joint,
                        friction,
                        damping,
                    )
                # elif joint.joint_type == "hinge":
                #     link_builder.set_joint_properties(
                #         "revolute",
                #         [[-np.inf, np.inf]],
                #         t_axis2parent,
                #         t_axis2joint,
                #         friction,
                #         damping,
                #     )
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

                import ipdb

                ipdb.set_trace()

        for child in body.findall("body"):
            self._parse_body(child, link_builder, defaults, builder)

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
            self._parse_body(body, None, self._root_default, builder)

        ### Parse contact and exclusions ###
        for contact in xml.findall("contact"):
            # self._parse_contact()

            pass

        ### Parse equality constraints ###

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

        articulations = []
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
