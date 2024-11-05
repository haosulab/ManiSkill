import numpy as np
import sapien
from transforms3d.euler import euler2quat, quat2euler

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.scene_builder.robocasa.utils.scene_utils import ROBOCASA_ASSET_DIR


class Box:
    def __init__(
        self,
        scene: ManiSkillScene,
        pos,
        size,
        name="box",
        texture="textures/wood/dark_wood_parquet.png",
        mat_attrib={"shininess": "0.1"},
        tex_attrib={"type": "cube"},
        rng=None,
        *args,
        **kwargs,
    ):
        self.name = name
        self.size = np.array(size)
        self.pos = np.array(pos)
        self.quat = [1, 0, 0, 0]
        self.scene = scene
        self.render_material = sapien.render.RenderMaterial()
        texture = str(ROBOCASA_ASSET_DIR / texture)
        self.render_material.base_color_texture = sapien.render.RenderTexture2D(
            filename=texture,
            mipmap_levels=1,
        )
        # for relative positioning
        self.origin_offset = np.array([0, 0, 0])
        self.scale = 1

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

    def set_pos(self, pos):
        self.pos = np.array(pos)

    def set_euler(self, euler):
        self.quat = euler2quat(*euler)

    @property
    def euler(self):
        return np.array(quat2euler(self.quat))

    def build(self, scene_idxs: list[int]):
        builder = self.scene.create_actor_builder()
        builder.set_scene_idxs(scene_idxs)
        builder.add_box_visual(half_size=self.size / 2, material=self.render_material)
        builder.add_box_collision(half_size=self.size / 2)
        builder.initial_pose = sapien.Pose(self.pos, self.quat)
        self.actor = builder.build_static(name=self.name + f"_{scene_idxs[0]}")
        return self

    @property
    def is_articulation(self):
        return False


class Wall:
    def __init__(
        self,
        scene: ManiSkillScene,
        name="wall",
        texture="textures/bricks/white_bricks.png",
        pos=None,
        quat=None,
        size=None,
        wall_side="back",
        mat_attrib={
            "texrepeat": "3 3",
            "reflectance": "0.1",
            "shininess": "0.1",
            "texuniform": "true",
        },
        tex_attrib={"type": "2d"},
        # parameters used for alignment
        backing=False,
        backing_extended=[False, False],
        default_wall_th=0.02,
        default_backing_th=0.1,
        rng=None,
        *args,
        **kwargs,
    ):
        self.mat_attrib = mat_attrib
        self.texture_repeat = [1, 1]
        if "texrepeat" in mat_attrib:
            self.texture_repeat = (
                np.fromstring(mat_attrib["texrepeat"], sep=" ", dtype=np.float32) / 2
            )
        self.tex_attrib = tex_attrib
        # change texture if used for backing
        self.backing = backing
        if backing:
            # texture = "textures/flat/light_gray.png"
            texture = "textures/flat/light_gray.png"
        # self.render_material = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1])
        self.render_material = sapien.render.RenderMaterial()
        texture = str(ROBOCASA_ASSET_DIR / texture)
        self.render_material.base_color_texture = sapien.render.RenderTexture2D(
            filename=texture,
            mipmap_levels=1,
        )

        self.wall_side = wall_side
        # set the rotation according to which side the wall is on
        if self.wall_side is not None:
            self.get_quat()

        # align everything to account for thickness & backing
        if self.wall_side == "floor":
            size[0] += default_wall_th * 2
            size[1] += default_wall_th * 2
            pos[2] -= size[2]
            if backing:
                pos[2] -= default_wall_th * 2
        else:
            size[0] += default_wall_th * 2
            shift = size[2] if not backing else size[2] + default_wall_th * 2
            self.shift = shift
            if self.wall_side == "left":
                pos[0] -= shift
            elif self.wall_side == "right":
                pos[0] += shift
            elif self.wall_side == "back":
                pos[1] += shift
            elif self.wall_side == "front":
                pos[1] -= shift

            if backing:
                size[1] += default_wall_th + default_backing_th
                pos[2] -= default_wall_th + default_backing_th

                # extend left/right side to form a perfect box
                if backing_extended[0]:
                    size[0] += default_backing_th
                    if self.wall_side in ["left", "right"]:
                        pos[1] += default_backing_th
                    else:
                        pos[0] -= default_backing_th
                if backing_extended[1]:
                    size[0] += default_backing_th
                    if self.wall_side in ["left", "right"]:
                        pos[1] -= default_backing_th
                    else:
                        pos[0] += default_backing_th
        self.name = name
        self.size = size
        self.pos = np.array(pos)
        self.scene = scene

    @property
    def is_articulation(self):
        return False

    def build(self, scene_idxs: list[int]):
        builder = self.scene.create_actor_builder()
        pos = self.pos
        if self.backing:
            return None
            builder.add_box_visual(half_size=self.size, material=self.render_material)
        else:
            builder.add_plane_repeated_visual(
                half_size=self.size[:2],
                mat=self.render_material,
                texture_repeat=self.texture_repeat,
            )
            if self.wall_side == "left":
                pos[0] += self.shift
            elif self.wall_side == "right":
                pos[0] -= self.shift
            elif self.wall_side == "back":
                pos[1] -= self.shift
            elif self.wall_side == "front":
                pos[1] += self.shift
        builder.add_box_collision(half_size=self.size)
        builder.initial_pose = sapien.Pose(pos, self.get_quat())
        builder.set_scene_idxs(scene_idxs)
        self.actor = builder.build_static(name=self.name + f"_{scene_idxs[0]}")
        return self

    def get_quat(self):
        """
        Returns the quaternion of the object based on the wall side

        Returns:
            list: quaternion
        """
        # quaternions are modified since the 2d repeated texture is flat and has a backface
        side_rots = {
            "back": [-0.707, 0.707, 0, 0],
            "front": [0, 0, 0.707, -0.707],
            "left": [0.5, 0.5, -0.5, -0.5],
            "right": [-0.5, 0.5, -0.5, 0.5],
            "floor": [0.707, 0, 0, 0.707],
        }
        if self.wall_side not in side_rots:
            raise ValueError()
        return side_rots[self.wall_side]


class Floor(Wall):
    def __init__(
        self,
        scene: ManiSkillScene,
        size,
        name="wall",
        texture="textures/bricks/red_bricks.png",
        mat_attrib={
            "texrepeat": "2 2",
            "texuniform": "true",
            "reflectance": "0.1",
            "shininess": "0.1",
        },
        *args,
        **kwargs,
    ):
        super().__init__(
            scene,
            size=size,
            name=name,
            texture=texture,
            wall_side="floor",
            mat_attrib=mat_attrib,
            *args,
            **kwargs,
        )
        self.name = name
        self.scene = scene

    def build(self, scene_idxs: list[int]):
        builder = self.scene.create_actor_builder()
        if self.backing:
            builder.add_box_visual(half_size=self.size, material=self.render_material)
        else:
            builder.add_plane_repeated_visual(
                pose=sapien.Pose(q=[0, 0, 1, 0]),
                half_size=self.size[:2],
                mat=self.render_material,
                texture_repeat=self.texture_repeat,
            )
            # Only ever add one plane collision
            if 0 in scene_idxs:
                builder.add_plane_collision(
                    pose=sapien.Pose(q=[0.7071068, 0, -0.7071068, 0])
                )
        builder.initial_pose = sapien.Pose(p=self.pos)
        builder.set_scene_idxs(scene_idxs)
        self.actor = builder.build_static(name=self.name + f"_{scene_idxs[0]}")
        return self


# class Floor(Wall):
#     def __init__(
#         self,
#         size,
#         name="wall",
#         texture="textures/bricks/red_bricks.png",
#         mat_attrib={
#             "texrepeat": "2 2",
#             "texuniform": "true",
#             "reflectance": "0.1",
#             "shininess": "0.1",
#         },
#         *args,
#         **kwargs
#     ):
#         # swap x, y axes due to rotation
#         size = [size[1], size[0], size[2]]

#         texture = xml_path_completion(texture, root=robocasa.models.assets_root)

#         # everything is the same except the plane is rotated to be horizontal
#         super().__init__(
#             name,
#             texture,
#             # horizontal plane
#             wall_side="floor",
#             size=size,
#             mat_attrib=mat_attrib,
#             *args,
#             **kwargs
#         )
