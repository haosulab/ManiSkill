import numpy as np
import sapien

from mani_skill import ASSET_DIR
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.scene_builder.robocasa.fixtures.wall import Wall


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
        **kwargs
    ):
        super().__init__(
            scene,
            size=size,
            name=name,
            texture=texture,
            mat_attrib=mat_attrib,
            *args,
            **kwargs
        )
        self.name = name
        self.scene = scene
        # import ipdb; ipdb.set_trace()
        self.size = np.array(size)
        # self.pos = np.array(pos)
        # self.render_material = sapien.render.RenderMaterial()
        # texture = str(ASSET_DIR / "scene_datasets/robocasa_dataset/assets" / texture)
        # if backing:
        #     texture = str(ASSET_DIR / "scene_datasets/robocasa_dataset/assets/textures/flat/light_gray.png")
        # self.render_material.base_color_texture = sapien.render.RenderTexture2D(
        #     filename=texture,
        #     mipmap_levels=1,
        # )

    def build(self):
        builder = self.scene.create_actor_builder()
        if self.backing:
            builder.add_box_visual(half_size=self.size, material=self.render_material)
        else:
            builder.add_repeated_2D_texture(
                half_size=self.size[:2], mat=self.render_material, texrepeat="3 3"
            )
            builder.add_plane_collision(
                pose=sapien.Pose(q=[0.7071068, 0, -0.7071068, 0])
            )
        # builder.add_plane_visual(scale=self.size * 2)

        builder.initial_pose = sapien.Pose(p=self.pos)
        # builder.initial_pose = sapien.Pose(self.pos, self.get_quat())
        self.actor = builder.build_static(name=self.name)
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
