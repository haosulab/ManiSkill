import os

import numpy as np
import sapien
import torch
from torch import Tensor
from transforms3d.euler import euler2quat

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.humanoid.scene_builders.dining_table import (
    DiningTableSceneBuilder,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


class UnitreeG1SetTableEnv(BaseEnv):
    """
    **Task Description:**
    A G1 humanoid robot must find the bowl and place it at the center of the place mat.

    **Randomizations:**
    - the place mat's xy position is randomized
    - the bowl's xy position is randomized

    **Success Conditions:**
    - the box is resting on top of the other table
    """

    SUPPORTED_REWARD_MODES = ["none"]
    SUPPORTED_ROBOTS = ["unitree_g1_simplified_upper_body_with_head_camera"]
    success_constraints = {}

    def __init__(
        self,
        *args,
        robot_uids="unitree_g1_simplified_upper_body_with_head_camera",
        **kwargs
    ):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22, max_rigid_patch_count=2**21
            ),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.6, -0.3, 1.3], [-0.4, 0.0, 0.75])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 3)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.3, 0.6, 1.3], [-0.4, 0.0, 0.75])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=np.pi / 3
        )

    def _load_lighting(self, options: dict):
        if self.scene_builder.builds_lighting:
            return
        return super()._load_lighting(options)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1]))

    def _load_scene(self, options: dict):
        self.scene_builder = DiningTableSceneBuilder(self)
        self.scene_builder.pose = sapien.Pose(p=[2.0, 0, 0])
        self.scene_builder.build()

        place_mat = self.scene.create_actor_builder()
        place_mat.add_visual_from_file(
            os.path.join(os.path.dirname(__file__), "assets/place_mat.glb")
        )
        place_mat.initial_pose = sapien.Pose([0, 0, 0.685163], [-0.5, 0.5, -0.5, 0.5])
        place_mat.build_kinematic("place_mat")

        bowl = self.scene.create_actor_builder()
        bowl.add_visual_from_file(
            os.path.join(os.path.dirname(__file__), "assets/bowl.glb")
        )
        bowl.add_multiple_convex_collisions_from_file(
            os.path.join(os.path.dirname(__file__), "assets/bowl.glb"),
            decomposition="coacd",
        )
        bowl.initial_pose = sapien.Pose(
            p=[0, 0, 0.714745], q=euler2quat(np.pi / 2, 0, np.pi / 2)
        )
        bowl.build("bowl")

    def _initialize_episode(self, env_idx: Tensor, options: dict):
        with torch.device(self.device):
            len(env_idx)
            self.scene_builder.initialize(env_idx, options)

    def evaluate(self) -> dict:
        for k, v in self.success_constraints.items():
            self.scene.actors[v["object_a"]].pose


@register_env("UnitreeG1SetTableEasy-v1", max_episode_steps=50)
class UnitreeG1SetTableEasyEnv(UnitreeG1SetTableEnv):
    pass
