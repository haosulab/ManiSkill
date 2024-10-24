from typing import List

import numpy as np
import sapien
import sapien.physx as physx
from transforms3d.euler import euler2quat

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.scene_builder.robocasa.utils.scene_utils import ROBOCASA_ASSET_DIR


class Window:
    """
    Window object. Supports creating windows with trim and glass. Proceduraly generated object (no xml needed)

    Args:
        name (str): name of the object

        size (list): size of the window(s) (considers multiple windows in the x-direction)

        ofs (list): offset of the window

        pos (list): position of the window

        quat (list): quaternion of the window

        window_bak (str): path to the window background texture

        texture (str): path to the window pane texture

        trim_th (float): thickness of the trim (depth/protrusion of the trim from the window pane)

        trim_size (float): size of the trim

        num_windows (int): number of windows placed side-by-side
    """

    def __init__(
        self,
        scene: ManiSkillScene,
        name,
        size,
        ofs=None,
        pos=None,
        quat=None,
        window_bak="textures/others/bk7.png",
        texture="textures/flat/white.png",
        trim_th=0.02,
        trim_size=0.015,
        num_windows=1,
        rng=None,
    ):
        self.scene = scene
        self.name = name
        self.objects: List[ActorBuilder] = []
        self.euler = [0, 0, 0]

        self.size = size
        self.origin_offset = [0, 0, 0]
        self.window_size = [size[0] / num_windows, size[1], size[2]]
        # self.texture = xml_path_completion(texture, robocasa.models.assets_root)
        self.texture = str(ROBOCASA_ASSET_DIR / texture)
        # self.window_bak = xml_path_completion(window_bak, robocasa.models.assets_root)
        self.window_bak = str(ROBOCASA_ASSET_DIR / window_bak)
        self.num_windows = num_windows
        self.pos = np.array([0, 0, 0]) if pos is None else np.array(pos)
        tex_attrib = {"type": "2d"}
        # now do not consider passed in quat bc havent combined
        # wanted quat with the rotation applied to the window
        # self.quats = quat
        self.quats = []
        self.trim_size = trim_size
        self.trim_th = trim_th
        trim_mat_attrib = {
            "texrepeat": "4 4",
            "specular": "0.1",
            "shininess": "0.1",
            "texuniform": "true",
        }

        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.1",
            "shininess": "0.1",
            "texuniform": "true",
        }
        self.trim_mat = sapien.render.RenderMaterial()
        self.trim_mat.set_base_color_texture(
            sapien.render.RenderTexture2D(
                filename=self.texture,
            )
        )
        self.window_mat = sapien.render.RenderMaterial()
        self.window_mat.set_base_color_texture(
            sapien.render.RenderTexture2D(
                filename=self.window_bak,
            )
        )
        # self.trim_mat = CustomMaterial(
        #     texture=self.texture,
        #     tex_name="panel_tex",
        #     mat_name="panel_mat",
        #     tex_attrib=tex_attrib,
        #     mat_attrib=trim_mat_attrib,
        #     shared=True,
        # )
        # self.window_mat = CustomMaterial(
        #     texture=self.window_bak,
        #     tex_name="blurred_bak",
        #     mat_name="window_mat",
        #     tex_attrib=tex_attrib,
        #     mat_attrib=mat_attrib,
        #     shared=True,
        # )

        self.center = np.array([0, 0, 0])
        self.scale = 1.0
        self.num_windows = num_windows
        self.ofs = ofs if ofs is not None else [0.0, 0.0, 0.0]
        self.ofs = np.array(self.ofs)

        self.create_window()

        # super().__init__(
        #     scene=scene,
        #     name=name,
        #     # objects=self.objects,
        #     # object_locations=self.positions,
        #     # object_quats=self.quats,
        #     # joints=None,
        # )

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()
        # change to create objects then create positions

    def build(self, scene_idxs: List[int]):
        self.actor_builder.set_scene_idxs(scene_idxs)
        self.actor_builder.initial_pose = sapien.Pose(
            p=np.array(self.pos), q=euler2quat(*self.euler)
        )
        self.actor = self.actor_builder.build_static(
            name=self.name + f"_{scene_idxs[0]}"
        )
        return self

    @property
    def is_articulation(self):
        return False

    def set_euler(self, euler):
        self.euler = euler

    def create_window(self):
        """
        Creates the window object with trim and glass by setting the relevant sizes and positions of the geoms in the window.
        This is very similar to the create panel functions in the cabinet panel classes
        """
        x, y, z = self.window_size
        x, y, z = x / 2, y / 2, z / 2
        # should change door to panel
        door_th = y - self.trim_th
        sizes = [
            [self.trim_size, self.trim_th, z],
            [self.trim_size, self.trim_th, z],
            [x - 2 * self.trim_size, self.trim_th, self.trim_size],
            [x - 2 * self.trim_size, self.trim_th, self.trim_size],
            [self.trim_size / 2, self.trim_th, z],
            [x - 2 * self.trim_size, self.trim_th, self.trim_size / 2],
        ] * self.num_windows

        # switch y and z sizes because we will rotate
        # we are doing this because textures are only displayed on the z-parallel part of the box
        sizes.append([x * self.num_windows, z, door_th])

        base_names = [
            "trim_left",
            "trim_right",
            "trim_top",
            "trim_bottom",
            "vert_trim",
            "horiz_trim",
        ]
        names = [f"{name}_{i}" for i in range(self.num_windows) for name in base_names]
        names.append("door")

        offsets = self._get_window_offsets()
        positions = []

        for offset in offsets:
            positions.extend(
                [
                    np.array([-x + self.trim_size + offset, -0.0045, 0]),
                    np.array([x - self.trim_size + offset, -0.0045, 0]),
                    np.array([offset, -0.0045, z - self.trim_size]),
                    np.array([offset, -0.0045, -z + self.trim_size]),
                    np.array([offset, -0.0045, 0]),
                    np.array([offset, -0.0045, 0]),
                ]
            )
        positions.append(np.array([0, 0, 0]))

        objects = []
        builder = self.scene.create_actor_builder()
        for i, (obj_name, size) in enumerate(zip(names, sizes)):
            obj_pose = sapien.Pose(p=positions[i] + self.ofs)
            if len(names) - 1 == i:
                obj_pose.q = [0, 0, 0.7071081, 0.7071055]
            if "door" in obj_name:
                # new_obj = BoxObject(
                #     name=obj_name, size=np.array(size), material=self.window_mat
                # )
                builder.add_box_collision(half_size=size, pose=obj_pose)
                builder.add_box_visual(
                    half_size=size,
                    name=obj_name,
                    material=self.window_mat,
                    pose=obj_pose,
                )
            else:
                # new_obj = BoxObject(
                #     name=obj_name, size=np.array(size), material=self.trim_mat
                # )
                builder.add_box_collision(half_size=size, pose=obj_pose)
                builder.add_box_visual(
                    half_size=size, name=obj_name, material=self.trim_mat, pose=obj_pose
                )
            # objects.append(builder)

        self.objects = objects
        self.actor_builder = builder
        self.positions = [position + self.ofs for position in positions]
        self.quats = [None] * (len(objects) - 1)
        self.quats.append([0, 0, 0.7071081, 0.7071055])

    def _get_window_offsets(self):
        """
        Gets x-direction offsets for the windows. This is relevant when creating multiple individual windows
        which are placed side-by-side
        """
        x = self.window_size[0] / 2
        start = (-self.size[0] / 2) + x
        end = self.size[0] / 2 - x
        offsets = np.linspace(start, end, self.num_windows)
        return offsets

    def set_pos(self, pos):
        """
        Set the position of the window

        Args:
            pos (list): position of the window
        """
        self.pos = np.array(pos)
        # self._obj.set("pos", a2s(pos))

    def update_state(self, env):
        return

    @property
    def nat_lang(self):
        return "windows"

    @property
    def rot(self):
        rot = s2a(self._obj.get("euler", "0.0 0.0 0.0"))
        return rot[2]


class FramedWindow(Window):
    """
    Window object with a frame around it

    Args:
        name (str): name of the object

        size (list): size of the window(s) (considers multiple windows in the x-direction)

        ofs (list): offset of the window

        pos (list): position of the window

        quat (list): quaternion of the window

        window_bak (str): path to the window background texture

        texture (str): path to the window pane texture

        trim_th (float): thickness of the trim (depth/protrusion of the trim from the window pane)

        trim_size (float): size of the trim

        num_windows (int): number of windows placed side-by-side

        frame_width (float): width of the frame around the window
    """

    def __init__(
        self,
        scene: ManiSkillScene,
        name,
        size,
        ofs=None,
        pos=None,
        quat=None,
        window_bak="textures/others/bk7.png",
        texture="textures/flat/white.png",
        trim_th=0.02,
        trim_size=0.015,
        num_windows=1,
        frame_width=0.05,
        rng=None,
    ):

        self.frame_width = frame_width
        super().__init__(
            scene=scene,
            name=name,
            size=size,
            ofs=ofs,
            pos=pos,
            quat=quat,
            window_bak=window_bak,
            texture=texture,
            trim_th=trim_th,
            trim_size=trim_size,
            num_windows=num_windows,
            rng=rng,
        )

    def create_window(self):
        """
        Creates the window object with trim and glass and frame by setting the relevant sizes and positions of the geoms in the window.
        """
        self.window_size = [
            (self.size[0] - self.frame_width) / self.num_windows,
            self.size[1],
            self.size[2] - self.frame_width,
        ]
        super().create_window()

        # created the window now add the frame!
        x, y, z = self.window_size
        x, y, z = x / 2, y / 2, z / 2
        sizes = [
            [self.size[0] / 2, y, self.frame_width / 4],
            [self.size[0] / 2, y, self.frame_width / 4],
            [self.frame_width / 4, y, z],
            [self.frame_width / 4, y, z],
        ]

        names = ["frame_top", "frame_bot", "frame_right", "frame_left"]
        val = (self.size[0] - self.frame_width) / 2
        frame_positions = [
            np.array([0, 0.00145, z + self.frame_width / 4]),
            np.array([0, 0.00145, -z - self.frame_width / 4]),
            np.array([val + self.frame_width / 4, 0.00145, 0]),
            np.array([-val - self.frame_width / 4, 0.00145, 0]),
        ]
        new_positions = [frame_pos + self.ofs for frame_pos in frame_positions]
        self.positions.extend(new_positions)
        self.quats.extend([None] * len(new_positions))

        for obj_name, size in zip(names, sizes):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=size)
            builder.add_box_visual(
                half_size=size, name=obj_name, material=self.trim_mat
            )
            self.objects.append(builder)

    def _get_window_offsets(self):
        """
        Gets x-direction offsets for the windows. This is relevant when creating multiple individual windows
        which are placed side-by-side. This is different than the super class because we need to account for the frame width
        """
        x = self.window_size[0] / 2
        start = ((self.frame_width - self.size[0]) / 2) + x
        end = ((self.size[0] - self.frame_width) / 2) - x
        offsets = np.linspace(start, end, self.num_windows)
        return offsets

    def set_pos(self, pos):
        """
        Set the position of the window

        Args:
            pos (list): position of the window

        """
        self.pos = np.array(pos)
        # self._obj.set("pos", a2s(pos))
