import numpy as np
import sapien

from mani_skill.utils.building import actors
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.scene_builder.table import TableSceneBuilder

def _build_by_type(builder: ActorBuilder, name, body_type):
    if body_type == "dynamic":
        actor = builder.build(name=name)
    elif body_type == "static":
        actor = builder.build_static(name=name)
    elif body_type == "kinematic":
        actor = builder.build_kinematic(name=name)
    else:
        raise ValueError(f"Unknown body type {body_type}")
    return actor

class CustomBuiltPrimitives:
    @staticmethod
    def build_cube(
        scene: ManiSkillScene,
        half_size: float,
        color,
        name: str,
        body_type: str = "dynamic",
        add_collision: bool = True,
        material: sapien.render.RenderMaterial = None,
        physics_properties: dict = {},

    ):
        builder = scene.create_actor_builder()

        if add_collision:
            if physics_properties:
                builder.add_box_collision(
                    half_size=[half_size] * 3,
                    density=physics_properties["density"]
                )
            else:
                builder.add_box_collision(
                    half_size=[half_size] * 3,
                )
        
        builder.add_box_visual(
            half_size=[half_size] * 3,
            material=sapien.render.RenderMaterial(base_color=color,) if material is None else material
        )
        
        return _build_by_type(builder, name, body_type)
    
    @staticmethod
    def build_red_white_target(
        scene: ManiSkillScene,
        radius: float,
        thickness: float,
        name: str,
        body_type: str = "dynamic",
        add_collision: bool = True,
    ):
        TARGET_RED = np.array([194, 19, 22, 255]) / 255
        builder = scene.create_actor_builder()
        builder.add_cylinder_visual(
            radius=radius,
            half_length=thickness / 2,
            material=sapien.render.RenderMaterial(base_color=TARGET_RED),
        )
        builder.add_cylinder_visual(
            radius=radius * 4 / 5,
            half_length=thickness / 2 + 1e-5,
            material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
        )
        builder.add_cylinder_visual(
            radius=radius * 3 / 5,
            half_length=thickness / 2 + 2e-5,
            material=sapien.render.RenderMaterial(base_color=TARGET_RED),
        )
        builder.add_cylinder_visual(
            radius=radius * 2 / 5,
            half_length=thickness / 2 + 3e-5,
            material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
        )
        builder.add_cylinder_visual(
            radius=radius * 1 / 5,
            half_length=thickness / 2 + 4e-5,
            material=sapien.render.RenderMaterial(base_color=TARGET_RED),
        )

        if add_collision:
            builder.add_cylinder_collision(
                radius=radius,
                half_length=thickness / 2,
            )
            builder.add_cylinder_collision(
                radius=radius * 4 / 5,
                half_length=thickness / 2 + 1e-5,
            )
            builder.add_cylinder_collision(
                radius=radius * 3 / 5,
                half_length=thickness / 2 + 2e-5,
            )
            builder.add_cylinder_collision(
                radius=radius * 2 / 5,
                half_length=thickness / 2 + 3e-5,
            )
            builder.add_cylinder_collision(
                radius=radius * 1 / 5,
                half_length=thickness / 2 + 4e-5,
            )
            
        return _build_by_type(builder, name, body_type)
    
    @staticmethod
    def build_red_white_target_V1(
        scene: ManiSkillScene,
        radius: float,
        thickness: float,
        name: str,
        body_type: str = "dynamic",
        add_collision: bool = True,
    ):
        TARGET_GREEN = np.array([19, 194, 22, 255]) / 255
        builder = scene.create_actor_builder()

        custom_material1 = sapien.render.RenderMaterial(
            base_color=TARGET_GREEN, 
            emission=[0.5]*4,
            metallic=1.0,
        )
        
        custom_material2 = sapien.render.RenderMaterial(
            base_color=[1, 1, 1, 1], 
            emission=[0.5]*4,
            metallic=1.0,
        )

        builder.add_cylinder_visual(
            radius=radius,
            half_length=thickness / 2,
            material=custom_material1,
        )
        
        builder.add_cylinder_visual(
            radius=radius * 4 / 5,
            half_length=thickness / 2 + 1e-5,
            material=custom_material2,
        )
        
        builder.add_cylinder_visual(
            radius=radius * 3 / 5,
            half_length=thickness / 2 + 2e-5,
            material=custom_material1,
        )
        
        builder.add_cylinder_visual(
            radius=radius * 2 / 5,
            half_length=thickness / 2 + 3e-5,
            material=custom_material2,
        )
        
        builder.add_cylinder_visual(
            radius=radius * 1 / 5,
            half_length=thickness / 2 + 4e-5,
            material=custom_material1,
        )

        if add_collision:
            builder.add_cylinder_collision(
                radius=radius,
                half_length=thickness / 2
            )
            builder.add_cylinder_collision(
                radius=radius * 4 / 5,
                half_length=thickness / 2 + 1e-5,
            )
            builder.add_cylinder_collision(
                radius=radius * 3 / 5,
                half_length=thickness / 2 + 2e-5,
            )
            builder.add_cylinder_collision(
                radius=radius * 2 / 5,
                half_length=thickness / 2 + 3e-5,
            )
            builder.add_cylinder_collision(
                radius=radius * 1 / 5,
                half_length=thickness / 2 + 4e-5,
            )
            
        return _build_by_type(builder, name, body_type)
