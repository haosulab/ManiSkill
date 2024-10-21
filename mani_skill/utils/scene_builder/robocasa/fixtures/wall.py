import sapien

from mani_skill import ASSET_DIR
from mani_skill.envs.scene import ManiSkillScene


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
        **kwargs
    ):
        # change texture if used for backing
        self.backing = backing
        if backing:
            # texture = "textures/flat/light_gray.png"
            texture = "textures/flat/light_gray.png"
        # self.render_material = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1])
        self.render_material = sapien.render.RenderMaterial()
        texture = str(ASSET_DIR / "scene_datasets/robocasa_dataset/assets" / texture)
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
        self.pos = pos
        self.scene = scene

    @property
    def is_articulation(self):
        return False

    def build(self):
        builder = self.scene.create_actor_builder()
        if self.backing:
            builder.add_box_visual(half_size=self.size, material=self.render_material)
        else:
            builder.add_repeated_2D_texture(
                half_size=self.size[:2], mat=self.render_material, texrepeat="3 3"
            )
        builder.add_box_collision(half_size=self.size)
        builder.initial_pose = sapien.Pose(self.pos, self.get_quat())
        self.actor = builder.build_static(name=self.name)
        return self

    def get_quat(self):
        """
        Returns the quaternion of the object based on the wall side

        Returns:
            list: quaternion
        """
        # quaternions are modified since the 2d repeated texture is flat and has a backface
        side_rots = {
            "back": [0.707, 0.707, 0, 0],
            "front": [0, 0, 0.707, -0.707],
            "left": [0.5, 0.5, 0.5, 0.5],
            "right": [0.5, -0.5, -0.5, 0.5],
            "floor": [0.707, 0, 0, 0.707],
        }
        if self.wall_side not in side_rots:
            raise ValueError()
        return side_rots[self.wall_side]
