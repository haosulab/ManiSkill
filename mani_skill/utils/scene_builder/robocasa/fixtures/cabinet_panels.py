import abc
import xml

import numpy as np
import sapien

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.scene_builder.robocasa.fixtures.handles import (
    BarHandle,
    BoxedHandle,
    KnobHandle,
)
from mani_skill.utils.scene_builder.robocasa.fixtures.mujoco_object import MujocoObject
from mani_skill.utils.scene_builder.robocasa.utils.scene_utils import ROBOCASA_ASSET_DIR


class CabinetPanel(MujocoObject):
    """
    Base class for cabinet panels which are attached to the cabinet body.

    Args:
        xml (str): Path to the xml file for the cabinet panel.

        size (list): Size of the cabinet panel in [w, d, h] format.

        name (str): Name of the cabinet panel.

        handle_type (str): Type of handle to attach to the cabinet panel.

        handle_config (dict): Configuration for the handle.

        handle_hpos (str): Horizontal position of the handle.

        handle_vpos (str): Vertical position of the handle.

        texture (str): Path to the texture file for the cabinet panel.
    """

    geom_names: list[str]
    """order of collision geometries loaded by the MJCF loader"""
    geom_names_set: set[str]
    material_overrides: dict[str, sapien.render.RenderMaterial] = {}

    def __init__(
        self,
        scene: ManiSkillScene,
        xml,
        size,  # format: [w, d, h]
        name,
        handle_type="bar",
        handle_config=None,
        handle_hpos=None,
        handle_vpos=None,
        texture=None,
    ):
        self.geom_names_set = set(self.geom_names)
        super().__init__(
            scene=scene,
            xml=xml,
            name=name,
            # joints=None,
            # duplicate_collision_geoms=True,
        )

        self.size = size
        self.texture = texture

        self.handle_type = handle_type
        self.handle_config = handle_config
        self.handle_hpos = handle_hpos
        self.handle_vpos = handle_vpos
        self._create_panel()
        self._set_texture()
        self._add_handle()

    def set_geom_dimensions(self, sizes, positions, rotated=False):
        if rotated:
            # rotation trick to make texture appear uniform
            # see .xml file
            for side in sizes.keys():
                # not the best detection method, TODO: check euler
                if "door" in side or "trim" in side:
                    sizes[side] = [sizes[side][i] for i in [0, 2, 1]]
        for i, ((part_name, size), (_, position)) in enumerate(
            zip(sizes.items(), positions.items())
        ):

            col_record = self.actor_builder.collision_records[
                self.geom_names.index(part_name)
            ]
            col_record.scale = size
            col_record.pose = sapien.Pose(p=position, q=col_record.pose.q)
            mat = self.loader._materials["mat"]
            if part_name in self.material_overrides:
                mat = self.material_overrides[part_name]
            self.actor_builder.add_box_visual(
                name=self.name + "_" + part_name,
                pose=col_record.pose,
                half_size=size,
                material=mat,
            )

    @abc.abstractmethod
    def _get_components(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _create_panel(self):
        pass

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

        if isinstance(inp, xml.etree.ElementTree.Element):
            if inp.tag == "texture":
                return True

        return False

    def _set_texture(self):
        """
        Set the texture for the cabinet panel.
        """
        if self.texture is None:
            return
        self.texture = str(ROBOCASA_ASSET_DIR / self.texture)
        for visual_record in self.actor_builder.visual_records:
            shortname = visual_record.name.replace(self.name + "_", "")
            if (
                shortname in self.geom_names_set
                and shortname not in self.material_overrides
            ):
                visual_record.material.base_color_texture = (
                    sapien.render.RenderTexture2D(
                        filename=self.texture,
                        mipmap_levels=1,
                    )
                )
        # texture = find_elements(
        #     self.root, tags="texture", attribs={"name": "tex"}, return_first=True
        # )
        # tex_is_2d = texture.get("type", None) == "2d"
        # tex_name = get_texture_name_from_file(self.texture)
        # if tex_is_2d:
        #     tex_name += "_2d"
        # texture.set("name", tex_name)
        # texture.set("file", self.texture)

        # material = find_elements(
        #     self.root,
        #     tags="material",
        #     attribs={"name": "{}_mat".format(self.name)},
        #     return_first=True,
        # )
        # material.set("name", "{}_mat".format(self.name))
        # material.set("texture", tex_name)

    def _add_handle(self):
        """
        Add a handle to the cabinet panel. Creates the handle object, positions it based off the handle_hpos and handle_vpos,
        and appends it to the cabinet panel body.
        """
        if self.handle_type is None:
            return
        elif self.handle_type == "bar":
            handle_class = BarHandle
            vpad = 0.20
            hpad = 0.05
        elif self.handle_type == "knob":
            handle_class = KnobHandle
            vpad = 0.05
            hpad = 0.05
        elif self.handle_type == "boxed":
            handle_class = BoxedHandle
            vpad = 0.20
            hpad = 0.05
        else:
            raise NotImplementedError

        panel_w = self.size[0]
        panel_h = self.size[2]

        handle = handle_class(
            scene=self.scene,
            name="{}_handle".format(self.name),
            panel_w=panel_w,
            panel_h=panel_h,
            **self.handle_config,
        )
        # handle_elem = handle.get_obj()

        if self.handle_vpos == "bottom":
            handle_z = -(panel_h / 2 - vpad)
        elif self.handle_vpos == "top":
            handle_z = panel_h / 2 - vpad
        elif self.handle_vpos == "center":
            handle_z = 0.0
        else:
            raise NotImplementedError

        if self.handle_hpos == "left":
            handle_x = -(panel_w / 2 - hpad)
        elif self.handle_hpos == "right":
            handle_x = panel_w / 2 - hpad
        elif self.handle_hpos == "center":
            handle_x = 0.0
        else:
            raise NotImplementedError
        for col_record, vis_record in zip(
            handle.actor_builder.collision_records, handle.actor_builder.visual_records
        ):
            col_record.pose = sapien.Pose(p=[handle_x, 0, handle_z]) * col_record.pose
            vis_record.pose = sapien.Pose(p=[handle_x, 0, handle_z]) * vis_record.pose
            self.actor_builder.collision_records.append(col_record)
            self.actor_builder.visual_records.append(vis_record)
        # handle_elem.set("pos", a2s([handle_x, 0, handle_z]))

        # parent_body = self.get_obj()

        # self.merge_assets(handle)
        # parent_body.append(handle_elem)


class SlabCabinetPanel(CabinetPanel):
    """
    Initialize a slab cabinet panel, which is a simple flat panel.
    """

    geom_names = ["door"]

    def __init__(self, *args, **kwargs):
        xml = "fixtures/cabinets/cabinet_panels/slab.xml"
        super().__init__(xml=xml, *args, **kwargs)

    # def _get_components(self):
    #     """
    #     Gets the geoms for the cabinet panel.
    #     """
    #     geom_names = ["door"]
    #     return self._get_elements_by_name(geom_names)[0]

    def _create_panel(self):
        """
        Creates the cabinet panel. This involves setting the size and position of the panel's geom
        """
        # geoms = self._get_components()

        # divide by 2 for mujoco convention
        x, y, z = [dim / 2 for dim in self.size]

        sizes = {"door": [x, y, z]}
        positions = {"door": [0, 0, 0]}
        self.set_geom_dimensions(sizes, positions, rotated=True)


class ShakerCabinetPanel(CabinetPanel):
    """
    Initialize a shaker cabinet panel, which is a simple flat panel with a trim.

    Args:
        trim_th (float): Thickness of the trim (depth).

        trim_size (float): Size of the trim (width/height).
    """

    geom_names = ["door", "trim_left", "trim_right", "trim_bottom", "trim_top"]

    def __init__(self, name, trim_th=0.02, trim_size=0.08, *args, **kwargs):
        self.trim_th = trim_th
        self.trim_size = trim_size

        xml = "fixtures/cabinets/cabinet_panels/shaker.xml"
        super().__init__(xml=xml, name=name, *args, **kwargs)

    # def _get_components(self):
    #     """
    #     Gets the geoms for the cabinet panel. This includes the door and the 4 sorrounding trims.
    #     """
    #     return self._get_elements_by_name(geom_names)[0]

    def _create_panel(self):
        """
        Creates the cabinet panel. This involves setting the size and position of the panel's door and trim geoms
        """
        # divide by 2 for mujoco convention
        x, y, z = self.size
        x, y, z = x / 2, y / 2, z / 2
        trim_th, trim_size = self.trim_th / 2, self.trim_size / 2

        # position door and trims such that (0, 0, 0) is center
        door_th = y - trim_th
        door_y = trim_th / 2
        trim_y = -door_th / 2

        sizes = {
            "door": [x - trim_size * 2, door_th, z - trim_size * 2],
            "trim_left": [trim_size, trim_th, z],
            "trim_right": [trim_size, trim_th, z],
            "trim_top": [x - 2 * trim_size, trim_th, trim_size],
            "trim_bottom": [x - 2 * trim_size, trim_th, trim_size],
        }
        positions = {
            "door": [0, door_y, 0],
            "trim_left": [-x + trim_size, trim_y, 0],
            "trim_right": [x - trim_size, trim_y, 0],
            "trim_top": [0, trim_y, z - trim_size],
            "trim_bottom": [0, trim_y, -z + trim_size],
        }

        # geoms = self._get_components()
        self.set_geom_dimensions(sizes, positions, rotated=True)


class RaisedCabinetPanel(CabinetPanel):
    """
    Initialize a raised cabinet panel, similar to the shaker panel, but with a raised door portion.

    Args:
        trim_th (float): Thickness of the trim (depth).

        trim_size (float): Size of the trim (width/height).

        raised_gap (float): Gap between the raised portion and the sorrounding trims
    """

    geom_names = [
        "door",
        "door_raised",
        "trim_left",
        "trim_right",
        "trim_bottom",
        "trim_top",
    ]

    def __init__(
        self, name, trim_th=0.02, trim_size=0.08, raised_gap=0.01, *args, **kwargs
    ):
        self.trim_th = trim_th
        self.trim_size = trim_size
        self.raised_gap = raised_gap

        xml = "fixtures/cabinets/cabinet_panels/raised.xml"
        super().__init__(xml=xml, name=name, *args, **kwargs)

    # def _get_components(self):
    #     """
    #     Gets the geoms for the cabinet panel. This includes the door, the 4 sorrounding trims, and the raised portion.
    #     """
    #     geom_names = [
    #         "door",
    #         "door_raised",
    #         "trim_left",
    #         "trim_right",
    #         "trim_bottom",
    #         "trim_top",
    #     ]
    # return self._get_elements_by_name(geom_names)[0]

    def _create_panel(self):
        """
        Creates the cabinet panel. This involves setting the size and position of the panel's door, trim, and raised portion geoms
        """

        # place the trims accordingly
        ShakerCabinetPanel._create_panel(self)

        x, y, z = self.size
        x, y, z = x / 2, y / 2, z / 2

        trim_size, trim_th = self.trim_size / 2, self.trim_th / 2
        raised_gap = self.raised_gap / 2

        # place and size the raised portion accordingly,
        sizes = {
            "door_raised": [
                x - 2 * trim_size - 2 * raised_gap,
                trim_th,
                z - 2 * trim_size - 2 * raised_gap,
            ]
        }
        positions = {"door_raised": [0, -(y - trim_th) / 2, 0]}

        # geoms = self._get_components()
        self.set_geom_dimensions(sizes, positions, rotated=True)


class DividedWindowCabinetPanel(CabinetPanel):
    """
    Initialize a divided window cabinet panel, which is a panel with a windowed door, trims around, and
    a 2 trims running down the middle of the window (horizontally and vertically aligned).

    Args:
        name (str): Name of the cabinet panel.

        trim_th (float): Thickness of the trim (depth).

        trim_size (float): Size of the trims (width/height).
    """

    geom_names = [
        "door",
        "trim_left",
        "trim_right",
        "trim_bottom",
        "trim_top",
        "horiz_trim",
        "vert_trim",
    ]

    def __init__(self, name, trim_th=0.02, trim_size=0.08, *args, **kwargs):

        self.trim_th = trim_th
        self.trim_size = trim_size

        xml = "fixtures/cabinets/cabinet_panels/divided_window.xml"
        super().__init__(xml=xml, name=name, *args, **kwargs)

    # def _get_components(self):
    #     """
    #     Gets the geoms for the cabinet panel. This includes the door, the 4 sorrounding trims,
    #     and the 2 trims running down the middle of the window.
    #     """
    #     geom_names = [
    #         "door",
    #         "trim_left",
    #         "trim_right",
    #         "trim_bottom",
    #         "trim_top",
    #         "horiz_trim",
    #         "vert_trim",
    #     ]
    #     return self._get_elements_by_name(geom_names)[0]

    def _create_panel(self):
        """
        Creates the cabinet panel. This involves setting the size and position of the panel's door, trims, and window trims
        """
        # divide by 2 for mujoco convention
        x, y, z = self.size
        x, y, z = x / 2, y / 2, z / 2
        trim_th, trim_size = self.trim_th / 2, self.trim_size / 2

        # position door and trims such that (0, 0, 0) is center
        door_th = y - trim_th
        door_y = trim_th / 2
        trim_y = -door_th / 2

        sizes = {
            "door": [x - trim_size * 2, door_th, z - trim_size * 2],
            "trim_left": [trim_size, trim_th, z],
            "trim_right": [trim_size, trim_th, z],
            "trim_top": [x - 2 * trim_size, trim_th, trim_size],
            "trim_bottom": [x - 2 * trim_size, trim_th, trim_size],
            "vert_trim": [trim_size / 3.5, trim_th, z],
            "horiz_trim": [x - 2 * trim_size, trim_th, trim_size / 3.5],
        }
        positions = {
            "door": [0, door_y, 0],
            "trim_left": [-x + trim_size, trim_y, 0],
            "trim_right": [x - trim_size, trim_y, 0],
            "trim_top": [0, trim_y, z - trim_size],
            "trim_bottom": [0, trim_y, -z + trim_size],
            "vert_trim": [0, trim_y, 0],
            "horiz_trim": [0, trim_y, 0],
        }

        # geoms = self._get_components()
        self.set_geom_dimensions(sizes, positions, rotated=True)


class FullWindowedCabinetPanel(CabinetPanel):
    """
    Initialize a full windowed cabinet panel, which is a panel with a windowed door and trims around.
    Same as the divided window panel but without the middle trims running through the window.

    Args:
        name (str): Name of the cabinet panel.

        trim_th (float): Thickness of the trim (depth).

        trim_size (float): Size of the trims (width/height).

        opacity (float): Opacity of the window. Defaults to 0.5 to create a "frosted" effect.
    """

    geom_names = ["door", "trim_left", "trim_right", "trim_bottom", "trim_top"]
    material_overrides: dict[str, sapien.render.RenderMaterial] = {}

    def __init__(
        self, name, trim_th=0.02, trim_size=0.08, opacity=0.5, *args, **kwargs
    ):
        self.trim_th = trim_th
        self.trim_size = trim_size
        self.opacity = opacity

        xml = "fixtures/cabinets/cabinet_panels/full_window.xml"
        super().__init__(xml=xml, name=name, *args, **kwargs)

    # def _get_components(self):
    #     """
    #     Gets the geoms for the cabinet panel. This includes the door and the 4 sorrounding trims.
    #     """
    #     geom_names = ["door", "trim_left", "trim_right", "trim_bottom", "trim_top"]
    #     return self._get_elements_by_name(geom_names)[0]

    def _create_panel(self):
        """
        Creates the cabinet panel. This involves setting the size and position of the panel's door and trim geoms
        """
        # place the trims accordingly
        self.material_overrides["door"] = self.loader._materials["transparent_material"]
        ShakerCabinetPanel._create_panel(self)


class CabinetShelf(MujocoObject):
    """
    Initialize a cabinet shelf, which is a simple flat panel but rotated 90 degrees.

    Args:
        size (list): Size of the cabinet shelf in [w, d, h] format.

        texture (str): Path to the texture file for the cabinet shelf.

        name (str): Name of the cabinet shelf.

        pos (list): Position of the cabinet shelf.
    """

    def __init__(
        self,
        scene,
        size,  # format: [w, d, h]
        texture,
        name,
        pos=None,
    ):
        super().__init__(
            scene=scene,
            xml="fixtures/cabinets/cabinet_panels/shelf.xml",
            name=name,
            # joints=None,
            # duplicate_collision_geoms=True,
        )

        self.size = np.array(size)
        self.pos = np.array(pos) if pos is not None else np.array([0, 0, 0])
        self._create_panel()

        self.texture = texture
        self._set_texture()

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

        if isinstance(inp, xml.etree.ElementTree.Element):
            if inp.tag == "texture":
                return True

        return False

    def _set_texture(self):
        """
        Set the texture for the cabinet shelf.
        """
        if self.texture is None:
            return

        self.texture = str(ROBOCASA_ASSET_DIR / self.texture)
        for visual_record in self.actor_builder.visual_records:
            visual_record.material.base_color_texture = sapien.render.RenderTexture2D(
                filename=self.texture,
                mipmap_levels=1,
            )

        # self.texture = xml_path_completion(
        #     self.texture, root=robocasa.models.assets_root
        # )
        # texture = find_elements(
        #     self.root, tags="texture", attribs={"name": "tex"}, return_first=True
        # )
        # tex_is_2d = texture.get("type", None) == "2d"
        # tex_name = get_texture_name_from_file(self.texture)
        # if tex_is_2d:
        #     tex_name += "_2d"
        # texture.set("name", tex_name)
        # texture.set("file", self.texture)

        # material = find_elements(
        #     self.root,
        #     tags="material",
        #     attribs={"name": "{}_mat".format(self.name)},
        #     return_first=True,
        # )
        # material.set("texture", tex_name)

    def _get_components(self):
        """
        Gets the geoms for the cabinet shelf.
        """
        geom_names = ["shelf"]
        return self._get_elements_by_name(geom_names)[0]

    def _create_panel(self):
        """
        Creates the cabinet shelf. This involves setting the size and position of the panel's geom
        """
        # geoms = self._get_components()

        sizes = {"shelf": self.size / 2}
        positions = {"shelf": self.pos}
        self.actor_builder.collision_records[0].scale = sizes["shelf"]
        self.actor_builder.collision_records[0].pose = sapien.Pose(
            p=positions["shelf"], q=self.actor_builder.collision_records[0].pose.q
        )
        self.actor_builder.add_box_visual(
            pose=self.actor_builder.collision_records[0].pose,
            half_size=sizes["shelf"],
            material=self.loader._materials["mat"],
        )
        # set_geom_dimensions(sizes, positions, geoms)
