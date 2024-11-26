import numpy as np
import sapien
import torch
from torch import Tensor

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.humanoid.scene_builders.dining_table import (
    DiningTableSceneBuilder,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("UnitreeG1PutAwayCup-v1", max_episode_steps=50)
class UnitreeG1PutAwayCupEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["unitree_g1_simplified_upper_body_with_head_camera"]

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
        pose = sapien_utils.look_at([1.0, 0.0, 1.6], [0, 0.0, 0.65])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 3)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.0, 0.0, 1.6], [0, 0.0, 0.65])
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
        self.scene_builder.build()

    def _initialize_episode(self, env_idx: Tensor, options: dict):
        with torch.device(self.device):
            len(env_idx)
            self.scene_builder.initialize(env_idx, options)
