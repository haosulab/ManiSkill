import abc
from xml.etree import ElementTree as ET

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.utils.scene_builder.robocasa.fixtures.mujoco_object import MujocoObject
from mani_skill.utils.scene_builder.robocasa.utils.scene_utils import ROBOCASA_ASSET_DIR


class Handle(MujocoObject):
    """
    Base class for all handles attached to cabinet/drawer panels

    Args:
        name (str): Name of the handle

        xml (str): Path to the xml file of the handle

        panel_w (float): Width of the panel to attach the handle to

        panel_h (float): Height of the panel to attach the handle to

        texture (str): Path to the texture file of the handle

        orientation (str): Orientation of the handle. Can be either "horizontal" (for drawers) or "vertical"

        length (float): Length of the handle
    """

    def __init__(
        self,
        scene,
        name,
        xml,
        panel_w,
        panel_h,
        texture="textures/metals/bright_metal.png",
        orientation="vertical",
        length=None,
    ):
        super().__init__(
            scene=scene,
            xml=xml,
            name=name,
            # joints=None,
            # duplicate_collision_geoms=True,
        )

        self.length = length
        self.orientation = orientation
        self.texture = str(ROBOCASA_ASSET_DIR / texture)
        # self.texture = xml_path_completion(texture, root=robocasa.models.assets_root)
        self.panel_w = panel_w
        self.panel_h = panel_h

        # for hinge cabinets
        self.side = None

        self._create_handle()
        self._set_texture()

    @abc.abstractmethod
    def _get_components(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _create_handle(self, positions, sizes):
        raise NotImplementedError

    def exclude_from_prefixing(self, inp):
        """
        Exclude all shared materials and their associated names from being prefixed.

        Args:
            inp (ET.Element or str): Element or its attribute to check for prefixing.

        Returns:
            bool: True if we should exclude the associated name(s) with @inp from being prefixed with naming_prefix
        """
        if "tex" in inp:
            return True

        if isinstance(inp, ET.Element):
            return inp.tag in ["texture"]

        return False

    def _set_texture(self):
        """
        Set the texture of the handle
        """
        if self.texture is None:
            return
        # set texture
        self.texture = str(ROBOCASA_ASSET_DIR / self.texture)
        for visual_record in self.actor_builder.visual_records:
            visual_record.material.base_color_texture = sapien.render.RenderTexture2D(
                filename=self.texture,
                mipmap_levels=1,
            )
        # texture = find_elements(
        #     self.root, tags="texture", attribs={"name": "tex"}, return_first=True
        # )
        # tex_name = get_texture_name_from_file(self.texture)
        # texture.set("file", self.texture)
        # texture.set("name", tex_name)

        # material = find_elements(
        #     self.root,
        #     tags="material",
        #     attribs={"name": "{}_mat".format(self.name)},
        #     return_first=True,
        # )
        # material.set("texture", tex_name)


class BarHandle(Handle):
    """
    Creates a bar handle

    Args:
        length (float): Length of the handle

        handle_pad (float): A minimum difference between handle length and cabinet panel height
    """

    def __init__(
        self,
        length=0.24,
        # connector_pad=0.05,
        handle_pad=0.04,
        *args,
        **kwargs
    ):
        # z-direction padding for top and bottom connectors
        # self.connector_pad = connector_pad
        # assert length > connector_pad * 2
        # z-direction padding for handle from sides of cabinet
        self.handle_pad = handle_pad

        super().__init__(
            xml="fixtures/handles/bar_handle.xml", length=length, *args, **kwargs
        )

    def _get_components(self):
        """
        Get the geoms of the handle
        """
        geom_names = ["handle", "handle_connector_top", "handle_connector_bottom"]
        body_names = []
        joint_names = []
        return self._get_elements_by_name(geom_names, body_names, joint_names)

    def _create_handle(self):
        """
        Calculates and sets and positions and sizes of each component of the handles
        Treats the three types of cabinets separately
        """

        # adjust handle size if necessary
        if self.panel_h < self.length + 2 * self.handle_pad:
            self.length = self.panel_h - 2 * self.handle_pad
            # if self.length < 3 * self.connector_pad:
            #     raise ValueError("Cabinet size {:.3f} is too small for " \
            #                      "bar handles.".format(self.panel_h))

        offset = self.length / 2 * 0.60  # - self.connector_pad
        # distance between main handle and door
        conn_len = 0.05

        # calculate positions for each component
        positions = {
            "handle": np.array([0, -conn_len, 0]),
            "handle_connector_top": np.array([0, -conn_len / 2, offset]),
            "handle_connector_bottom": np.array([0, -conn_len / 2, -offset]),
        }
        sizes = {
            "handle": [0.013, self.length / 2],
            "handle_connector_top": [0.008, conn_len / 2],
            "handle_connector_bottom": [0.008, conn_len / 2],
        }
        eulers = {}

        if self.orientation == "horizontal":
            positions["handle_connector_top"][[0, 2]] = positions[
                "handle_connector_top"
            ][[2, 0]]
            positions["handle_connector_bottom"][[0, 2]] = positions[
                "handle_connector_bottom"
            ][[2, 0]]
            # compared to original code this swaps xy due to convention difference
            eulers["handle"] = [1.5708, 0, 0]

        # geoms, bodies, joints = self._get_components()
        for i, side in enumerate(positions.keys()):
            pos = positions[side]
            size = sizes[side]
            quat = self.actor_builder.collision_records[i].pose.q
            if side in eulers:
                quat = euler2quat(*eulers[side])
            self.actor_builder.add_cylinder_visual(
                pose=sapien.Pose(p=pos, q=quat),
                radius=size[0],
                half_length=size[1],
                material=self.loader._materials["mat"],
                name=side,
            )
            self.actor_builder.collision_records[i].radius = size[0]
            self.actor_builder.collision_records[i].length = size[1]
            self.actor_builder.collision_records[i].pose = sapien.Pose(p=pos, q=quat)
            # for geom in geoms[side]:
            #     if geom is None:
            #         continue
            #     geom.set("pos", a2s(positions[side]))
            #     geom.set("size", a2s(sizes[side]))

            #     if eulers.get(side) is not None:
            #         geom.set("euler", a2s(eulers[side]))


class BoxedHandle(Handle):
    """
    Creates a boxed handle

    Args:
        length (float): Length of the handle

        handle_pad (float):  A minimum difference between handle length and cabinet panel height
    """

    geom_names = set(["handle", "handle_connector_top", "handle_connector_bottom"])

    def __init__(self, length=0.24, handle_pad=0.04, *args, **kwargs):
        self.handle_pad = handle_pad
        super().__init__(
            xml="fixtures/handles/boxed_handle.xml", length=length, *args, **kwargs
        )

    # def _get_components(self):
    #     """
    #     Get the geoms of the handle
    #     """
    #     geom_names = ["handle", "handle_connector_top", "handle_connector_bottom"]
    #     body_names = []
    #     joint_names = []
    #     return self._get_elements_by_name(geom_names, body_names, joint_names)

    def _create_handle(self):
        """
        Calculates and sets and positions and sizes of each component of the handles
        Treats the three types of cabinets separately
        """

        # adjust handle size if necessary
        if self.panel_h < self.length + 2 * self.handle_pad:
            self.length = self.panel_h - 2 * self.handle_pad

        conn_len = 0.05
        connector_depth = (conn_len / 2) - 0.01
        connector_zpos = (self.length / 2) - 0.01

        # calculate positions for each component
        positions = {
            "handle": np.array([0, -conn_len, 0]),
            "handle_connector_top": np.array([0, -conn_len / 2, connector_zpos]),
            "handle_connector_bottom": np.array([0, -conn_len / 2, -connector_zpos]),
        }
        sizes = {
            "handle": [0.01, 0.01, self.length / 2],
            "handle_connector_top": [0.01, 0.01, connector_depth],
            "handle_connector_bottom": [0.01, 0.01, connector_depth],
        }
        eulers = {}

        if self.orientation == "horizontal":
            positions["handle_connector_top"][[0, 2]] = positions[
                "handle_connector_top"
            ][[2, 0]]
            positions["handle_connector_bottom"][[0, 2]] = positions[
                "handle_connector_bottom"
            ][[2, 0]]
            eulers["handle"] = [0, 1.5708, 0]
        for i, side in enumerate(positions.keys()):
            pos = positions[side]
            size = sizes[side]
            quat = self.actor_builder.collision_records[i].pose.q
            if side in eulers:
                quat = euler2quat(*eulers[side])
            self.actor_builder.add_box_visual(
                pose=sapien.Pose(p=pos, q=quat),
                half_size=size,
                material=self.loader._materials["mat"],
                name=side,
            )
            self.actor_builder.collision_records[i].scale = size
            self.actor_builder.collision_records[i].pose = sapien.Pose(p=pos, q=quat)
        # geoms, bodies, joints = self._get_components()
        # for side in positions.keys():
        #     for geom in geoms[side]:
        #         if geom is None:
        #             continue
        #         geom.set("pos", a2s(positions[side]))
        #         geom.set("size", a2s(sizes[side]))

        #         if eulers.get(side) is not None:
        #             geom.set("euler", a2s(eulers[side]))


class KnobHandle(Handle):
    """
    Creates a knob handle
    """

    def __init__(self, handle_pad=0.07, *args, **kwargs):
        super().__init__(
            xml="fixtures/handles/knob_handle.xml",
            # length=length,
            *args,
            **kwargs
        )

        # z-direction padding for handle from sides of cabinet
        self.handle_pad = handle_pad

    def _get_components(self):
        """
        Get the geoms of the handle
        """
        geom_names = ["handle"]
        body_names = []
        joint_names = []
        return self._get_elements_by_name(geom_names, body_names, joint_names)

    def _create_handle(self):
        """
        Calculates and sets and positions and sizes of each component of the handles
        """

        # calculate positions for each component
        positions = {
            "handle": np.array([0, -0.017, 0]),
        }
        # radius, depth
        sizes = {
            "handle": [0.015, 0.017],
        }

        # geoms, bodies, joints = self._get_components()
        # for side in positions.keys():
        #     for geom in geoms[side]:
        #         if geom is None:
        #             continue
        #         geom.set("pos", a2s(positions[side]))
        #         geom.set("size", a2s(sizes[side]))
        for i, side in enumerate(positions.keys()):
            col_record = self.actor_builder.collision_records[i]
            col_record.pose = sapien.Pose(p=positions[side], q=col_record.pose.q)
            self.actor_builder.add_cylinder_visual(
                pose=sapien.Pose(p=positions[side], q=col_record.pose.q),
                radius=sizes[side][0],
                half_length=sizes[side][1],
                material=self.loader._materials["mat"],
                name=side,
            )
            col_record.radius = sizes[side][0]
            col_record.length = sizes[side][1]
