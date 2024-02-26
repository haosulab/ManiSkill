import os.path as osp
from pathlib import Path

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
from sapien import Pose, physx
from transforms3d import euler

from mani_skill2 import ASSET_DIR
from mani_skill2.utils.building.actors import build_actor_ai2
from mani_skill2.utils.building.articulations import (
    build_articulation_from_file as build_articulation_from_file_core,  # TODO (stao): remove this temp code
)
from mani_skill2.utils.geometry.trimesh_utils import (
    get_articulation_meshes,
    merge_meshes,
)
from mani_skill2.utils.scene_builder import SceneBuilder

KITCHEN_ASSET_DIR = Path(osp.dirname(__file__), "assets")


def build_articulation_from_file(scene, model_id, **kwargs):
    return build_articulation_from_file_core(
        scene,
        str(ASSET_DIR / "partnet_mobility/dataset" / f"{model_id}/mobility.urdf"),
        **kwargs,
    )


def build_wall(scene: sapien.Scene, name):
    wall_mtl = sapien.render.RenderMaterial()
    wall_mtl.base_color = [32 / 255, 67 / 255, 80 / 255, 1]
    wall_mtl.metallic = 0.0
    wall_mtl.roughness = 0.9
    wall_mtl.specular = 0.1
    wall_mtl.diffuse_texture = sapien.render.RenderTexture2D(
        str(KITCHEN_ASSET_DIR / "brick_wall_02_diff_1k.jpg")
    )
    # Add some walls
    wall_1 = scene.create_actor_builder()
    wall_1.add_visual_from_file(
        str(KITCHEN_ASSET_DIR / "tiled_wall.obj"),
        scale=(3, 3, 2),  # kitchens are usually about 2.5 to 3 meters tall
        material=wall_mtl,
        pose=Pose(q=euler.euler2quat(0, np.pi / 2, 0)),
    )
    wall_1.add_convex_collision_from_file(
        str(KITCHEN_ASSET_DIR / "tiled_wall.obj"),
        scale=(3, 3, 2),
        pose=Pose(q=euler.euler2quat(0, np.pi / 2, 0)),
    )
    wall_1 = wall_1.build_static(name=name)
    return wall_1


class KitchenSceneBuilder(SceneBuilder):
    def build(self, scene: sapien.Scene):
        refrigerator, _ = build_articulation_from_file(scene, "10620", scale=1)
        refrigerator.set_pose(
            refrigerator.pose
            * Pose([-1.2, -0.993, 0], q=euler.euler2quat(0, 0, -np.pi / 2))
        )
        cabinet_1, bounds1 = build_articulation_from_file(scene, "45194", scale=0.65)
        cabinet_1.set_pose(cabinet_1.pose * Pose([0, 0, 0]))
        cabinet_2, bounds2 = build_articulation_from_file(
            scene, "45203", scale=0.490344571
        )
        cabinet_2.set_pose(cabinet_2.pose * Pose([0, -0.585, 0]))

        zero_height = bounds1[1, 2] - bounds1[0, 2]

        # objects on front wall
        oven, _ = build_articulation_from_file(scene, "101931", scale=0.72)
        oven.set_pose(oven.pose * Pose([0, 0.81, 0]))
        dishwasher, _ = build_articulation_from_file(scene, "11622", scale=0.52)
        dishwasher.set_pose(dishwasher.pose * Pose([0, 1.5, 0]))

        bin = build_actor_ai2("bin_21", scene, name="bin", kinematic=True)
        bin.set_pose(bin.pose * Pose([0, 2.2, 0]))

        # objects on counters
        toaster, _ = build_articulation_from_file(scene, "103485", scale=0.2)
        toaster.set_pose(
            toaster.pose * Pose([0.1, -0.34, 0], q=euler.euler2quat(0, 0, -np.pi / 2))
        )
        microwave, _ = build_articulation_from_file(scene, "7167", scale=0.4)
        microwave.set_pose(
            microwave.pose * Pose([-0.12, -1, 0], q=euler.euler2quat(0, 0, -np.pi / 2))
        )

        apple = build_actor_ai2("Apple_3", scene, name="bin", kinematic=False)
        apple.set_pose(apple.pose * Pose([0.2, 0.1, 0]))
        bowl_1 = build_actor_ai2("Bowl_24", scene, name="bowl_1", kinematic=False)
        bowl_1.set_pose(bowl_1.pose * Pose([0, 0.12, 0]))
        fork_1 = build_actor_ai2("Fork_1", scene, name="fork_1", kinematic=False)
        fork_1.set_pose(
            Pose(
                [0.0236264, 0.238803, 0.00920353],
                [0.771614, 0.0036494, -0.00147362, 0.636079],
            )
        )

        # other decor objects
        painting_1 = build_actor_ai2(
            "Wall_Decor_Painting_2", scene, name="painting_1", kinematic=True
        )
        painting_1.set_pose(
            painting_1.pose
            * Pose([0.3, -0.2, 0.9], q=euler.euler2quat(0, 0, -np.pi / 2))
        )

        wine_bottle = build_actor_ai2(
            "Wine_Bottle_1", scene, name="wine_bottle", kinematic=False
        )
        wine_bottle.set_pose(wine_bottle.pose * Pose([0, 0.35, 0]))

        plant_1 = build_actor_ai2(
            "Houseplant_24", scene, name="plant_1", kinematic=False
        )
        plant_1.set_pose(
            plant_1.pose
            * Pose([-0.12, -1, 0.3775], q=euler.euler2quat(0, 0, -np.pi / 2))
        )

        pot_1 = build_actor_ai2("Pot_16", scene, name="pot_1", kinematic=False)
        pot_1.set_pose(pot_1.pose * Pose([-0.321, -0.927, -0.3375]))

        # add cabinet counter tops
        counter_tops = []
        counter_top_thickness = 0.007
        for cabinet in [cabinet_1, cabinet_2]:
            rend_mtl = sapien.render.RenderMaterial()
            rend_mtl.base_color = [0.95, 0.89, 0.92, 1]
            rend_mtl.metallic = 0.0
            rend_mtl.roughness = 0.2
            rend_mtl.specular = 0.9
            bounds = merge_meshes(get_articulation_meshes(cabinet)).bounds
            cabinet_counter_top_1 = scene.create_actor_builder()
            counter_top_half_size = [
                (bounds[1, 0] - bounds[0, 0]) / 2,
                (bounds[1, 1] - bounds[0, 1]) / 2,
                counter_top_thickness,
            ]
            cabinet_counter_top_1.add_box_visual(
                Pose(),
                counter_top_half_size,
                rend_mtl,
                "",
            )
            cabinet_counter_top_1.add_box_collision(
                Pose(),
                counter_top_half_size,
                None,
            )
            cabinet_counter_top_1 = cabinet_counter_top_1.build_static()
            cabinet_counter_top_1.set_pose(
                Pose(p=[cabinet.pose.p[0], cabinet.pose.p[1], 0])
            )
            counter_tops.append(cabinet_counter_top_1)
        rend_mtl = sapien.render.RenderMaterial()
        rend_mtl.base_color = [0.95, 0.89, 0.92, 1]
        rend_mtl.metallic = 0.0
        rend_mtl.roughness = 0.2
        rend_mtl.specular = 0.9
        cabinet_counter_top_corner = scene.create_actor_builder()
        counter_top_half_size = [0.48, 0.35, counter_top_thickness]
        cabinet_counter_top_corner.add_box_visual(
            Pose(),
            counter_top_half_size,
            rend_mtl,
            "",
        )
        cabinet_counter_top_corner.add_box_collision(
            Pose(),
            counter_top_half_size,
            None,
        )
        cabinet_counter_top_corner = cabinet_counter_top_corner.build_static(
            name="cabinet_counter_top_corner"
        )
        cabinet_counter_top_corner.set_pose(
            cabinet_counter_top_corner.pose * Pose([-0.18, -1.05, 0])
        )

        def create_wood_panel(size, thickness, name, wood_type="dark"):
            rend_mtl = sapien.render.RenderMaterial()
            rend_mtl.base_color = [0.95, 0.89, 0.92, 1]
            rend_mtl.metallic = 0.0
            rend_mtl.roughness = 0.2
            rend_mtl.specular = 0.9
            if wood_type == "dark":
                rend_mtl.diffuse_texture = sapien.render.RenderTexture2D(
                    str(KITCHEN_ASSET_DIR / "wood_texture_1.jpg"),
                )
            elif wood_type == "light":
                rend_mtl.diffuse_texture = sapien.render.RenderTexture2D(
                    str(KITCHEN_ASSET_DIR / "wood_texture_0.jpg"),
                )
            builder = scene.create_actor_builder()
            half_size = [size[0] / 2, size[1] / 2, thickness]
            builder.add_box_visual(
                Pose(),
                [size[0] / 2, size[1] / 2, thickness],
                rend_mtl,
            )
            builder.add_box_collision(
                Pose(),
                half_size,
                None,
            )
            return builder.build_static(name=name)

        counter_inside_panel_1 = create_wood_panel(
            [0.8, 0.65], counter_top_thickness * 2, "counter_inside_panel_1"
        )
        counter_inside_panel_1.set_pose(
            Pose(p=[-0.634, -1.05, -0.409], q=euler.euler2quat(0, np.pi / 2, 0))
        )
        counter_inside_panel_2 = create_wood_panel(
            [0.8, 0.65], counter_top_thickness * 2, "counter_inside_panel_2"
        )
        counter_inside_panel_2.set_pose(
            Pose(p=[0.262, -1.05, -0.409], q=euler.euler2quat(0, np.pi / 2, 0))
        )

        counter_inside_panel_3 = create_wood_panel(
            [0.98, 0.12], counter_top_thickness * 2, "counter_inside_panel_3"
        )
        counter_inside_panel_3.set_pose(
            Pose(p=[-0.145, -0.75, -0.74], q=euler.euler2quat(np.pi / 2, 0, np.pi))
        )
        counter_inside_shelf_1 = create_wood_panel(
            [0.90, 0.65],
            counter_top_thickness,
            "counter_inside_shelf_1",
            wood_type="light",
        )
        counter_inside_shelf_1.set_pose(
            Pose(p=[-0.19, -1.07, -0.69], q=euler.euler2quat(0, 0, 0))
        )
        counter_inside_shelf_2 = create_wood_panel(
            [0.90, 0.65],
            counter_top_thickness,
            "counter_inside_shelf_2",
            wood_type="light",
        )
        counter_inside_shelf_2.set_pose(
            Pose(p=[-0.19, -1.07, -0.35], q=euler.euler2quat(0, 0, 0))
        )

        wall_1 = build_wall(scene=scene, name="wall_1")
        wall_1.set_pose(Pose(p=[0.295, 0, 1.5]))
        wall_2 = build_wall(scene=scene, name="wall_2")
        wall_2.set_pose(Pose(p=[0.295, 3, 1.5]))
        wall_3 = build_wall(scene=scene, name="wall_3")
        wall_3.set_pose(Pose(p=[-1, -1.37, 1.5], q=euler.euler2quat(0, 0, -np.pi / 2)))
        wall_4 = build_wall(scene=scene, name="wall_4")
        wall_4.set_pose(Pose(p=[-2.5, 0, 1.5], q=euler.euler2quat(0, 0, -np.pi)))
        wall_5 = build_wall(scene=scene, name="wall_5")
        wall_5.set_pose(Pose(p=[-2.5, 3, 1.5], q=euler.euler2quat(0, 0, -np.pi)))
        wall_6 = build_wall(scene=scene, name="wall_6")
        wall_6.set_pose(Pose(p=[-1, 4.5, 1.5], q=euler.euler2quat(0, 0, np.pi / 2)))

        objects_fixed_on_floor = [
            refrigerator,
            cabinet_1,
            cabinet_2,
            bin,
            dishwasher,
            wall_1,
            wall_2,
            wall_3,
            wall_4,
            wall_5,
            wall_6,
            oven,
        ]
        # move objects on to the floor, which is < 0 on the z-axis as z = 0 means the surface in front of the robot in the kitchen.
        for obj in objects_fixed_on_floor:
            obj.set_pose(obj.pose * Pose([0, 0, -zero_height - counter_top_thickness]))

        altitude = -zero_height
        # add the roof
        rend_mtl = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1])
        builder = scene.create_actor_builder()
        builder.add_plane_visual(
            scale=(2, 6 / 2, 3 / 2),
            pose=Pose(
                p=[0, 0, altitude + 3 - 1e-2], q=euler.euler2quat(0, np.pi / 2, 0)
            ),
            material=rend_mtl,
        )
        builder.add_plane_collision(
            Pose(p=[0, 0, altitude + 3 - 1e-2], q=euler.euler2quat(0, np.pi / 2, 0)),
        )
        builder.set_physx_body_type("static")
        roof = builder.build()
        roof.set_pose(Pose([(0.295 - 2.5) / 2, (4.5 - 1.37) / 2, 0]))
        roof.name = "roof"

        # add the ground
        rend_mtl = sapien.render.RenderMaterial()
        rend_mtl.diffuse_texture = sapien.render.RenderTexture2D(
            str(KITCHEN_ASSET_DIR / "laminate_floor_02_diff_1k.jpg")
        )

        builder = scene.create_actor_builder()
        builder.add_visual_from_file(
            str(KITCHEN_ASSET_DIR / "tiled_floor.obj"),
            scale=(3, 6, 2),
            pose=Pose(p=[0, 0, altitude], q=euler.euler2quat(0, np.pi, 0)),
            material=rend_mtl,
        )
        builder.add_plane_collision(
            Pose(p=[0, 0, altitude], q=[0.7071068, 0, -0.7071068, 0]),
        )
        builder.set_physx_body_type("static")
        ground = builder.build()
        ground.set_pose(Pose([(0.295 - 2.5) / 2, (4.5 - 1.37) / 2, 0]))
        ground.name = "ground"
        shadow = True  # self.enable_shadow
        scene.set_ambient_light([0.3, 0.3, 0.3])

        # Only the first of directional lights can have shadow
        # scene.add_directional_light(
        #     [1, 1, -1], [1, 1, 1], shadow=shadow, shadow_scale=5, shadow_map_size=2048, position=[0, 0, 1]
        # )
        def add_ceiling_lamp(position):
            builder = scene.create_actor_builder()
            # rend_mtl = sapien.render.RenderMaterial(base_color=[1,1,1,1], specular=1, metallic=1, roughness=0.5)
            # rend_mtl.emission =[228/255,112/255,37/255, 1]
            # builder.add_sphere_visual(Pose(p=[position[0], position[1], position[2]]), radius=0.15, material=rend_mtl)
            builder.build_static()
            scene.add_point_light(
                color=[1, 1, 1],
                shadow=shadow,
                position=[position[0], position[1], position[2] - 0.6],
            )
            # scene.add_point_light(
            #     direction=[0, 0, -1], color=[1, 1, 1], shadow=shadow, position=[position[0], position[1], position[2]-0.2]
            # )

        add_ceiling_lamp([-1, 3, 2])
        add_ceiling_lamp([-1, 0, 2])

        return
