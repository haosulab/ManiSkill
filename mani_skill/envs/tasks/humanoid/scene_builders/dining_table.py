import copy
import os
from pathlib import Path
from typing import List

import numpy as np
import sapien
from torch import Tensor
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.unitree_g1.g1_upper_body import UnitreeG1UpperBody
from mani_skill.utils.building import ground
from mani_skill.utils.scene_builder.scene_builder import SceneBuilder


class DiningTableSceneBuilder(SceneBuilder):
    builds_lighting = True
    pose: sapien.Pose = None
    """pose to transform entire scene by"""
    scene_scale = 1.0
    """scale of the scene. Depending on robot this should be adjusted for the robot's height"""
    furnished = False
    """whether to add furniture to the scene"""

    def build(self, build_config_idxs: List[int] = None):
        scale = [self.scene_scale] * 3
        if self.pose is not None:
            scene_pose = self.pose
        else:
            scene_pose = sapien.Pose()
        floor = self.scene.create_actor_builder()
        floor.add_plane_collision(pose=sapien.Pose(q=[0.7071068, 0, -0.7071068, 0]))
        floor.initial_pose = sapien.Pose() * scene_pose
        self.floor = floor.build_static(name="floor")

        wall = self.scene.create_actor_builder()
        # wall.add_nonconvex_collision_from_file(filename=os.path.join(os.path.dirname(__file__), "../assets/walls.obj"), pose=sapien.Pose(p=[0, 0, 0], q=euler2quat(np.pi / 2, 0, 0)), scale=scale)
        wall.add_visual_from_file(
            filename=os.path.join(os.path.dirname(__file__), "../assets/walls.obj"),
            pose=sapien.Pose(p=[0, 0, 0], q=euler2quat(np.pi / 2, 0, 0)),
            scale=scale,
        )
        wall.initial_pose = sapien.Pose() * scene_pose
        self.wall = wall.build_static(name="wall")

        window = self.scene.create_actor_builder()
        window_mat = sapien.render.RenderMaterial(
            metallic=0,
            transmission=1.0,
            roughness=0.050,
            ior=1.3,
            base_color=[1, 1, 1, 0.3],
        )
        window.add_box_visual(half_size=[0.02, 0.8, 0.8], material=window_mat)
        window.initial_pose = sapien.Pose(p=[-2.98, 0, 1.2]) * scene_pose
        self.window = window.build_static(name="window")

        model_dir = Path(
            os.path.join(
                os.path.dirname(__file__),
                "../../../../utils/scene_builder/table/assets",
            )
        )
        table_model_file = str(model_dir / "table.glb")
        # table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, scale[2] * 1.3 * 0.525510228 / 2]),
            half_size=(
                scale[0] * 0.9 * 0.690857142 / 2,
                scale[1] * 0.9 * 1.381714286 / 2,
                scale[2] * 1.3 * 0.525510228 / 2,
            ),
        )
        builder.add_visual_from_file(
            filename=table_model_file,
            scale=[scale[0] * 0.9, scale[1] * 0.9, scale[2] * 1.3],
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        self.table = builder.build_static(name="table")

        # build lighting
        self.scene.add_directional_light(
            [1, 0.2, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=2048
        )
        self.scene.set_environment_map(
            os.path.join(
                os.path.dirname(__file__), "../assets/sunset_jhbcentral_4k.exr"
            )
        )
        self.scene.add_area_light_for_ray_tracing(
            pose=sapien.Pose([-0.310871, 0, 2.4], [0.707107, 0, 0.707107, 0]),
            half_width=1.5 * self.scene_scale,
            half_height=3 * self.scene_scale,
            color=[1, 1, 1],
        )

        if (
            self.env.robot_uids == "unitree_g1_simplified_upper_body_with_head_camera"
            or self.env.robot_uids == "unitree_g1_simplified_upper_body"
        ):
            self.robot_init_pose = copy.deepcopy(
                UnitreeG1UpperBody.keyframes["standing"].pose
            )
            self.robot_init_qpos = copy.deepcopy(
                UnitreeG1UpperBody.keyframes["standing"].qpos
            )
            self.robot_init_pose.p = [-0.4, 0, 0.755]

    def initialize(self, env_idx: Tensor, init_config_idxs: List[int] = None):
        self.env.agent.robot.pose = self.robot_init_pose
        self.env.agent.robot.qpos = self.robot_init_qpos
