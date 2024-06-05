import numpy as np
import torch
from mani_skill.envs.tasks.control.cartpole import CartpoleBalanceEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SceneConfig, SimConfig

@register_env("CartpoleBalanceBenchmark-v1")
class CartPoleBalanceBenchmarkEnv(CartpoleBalanceEnv):
    def __init__(self, *args, camera_width=128, camera_height=128, num_cameras=1, **kwargs):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.num_cameras = num_cameras
        super().__init__(*args, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=120,
            spacing=20,
            control_freq=60,
            scene_cfg=SceneConfig(
                bounce_threshold=0.5,
                solver_position_iterations=4, solver_velocity_iterations=0
            ),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0, -4, 1], target=[0, 0, 1])
        sensor_configs = []
        if self.num_cameras is not None:
            for i in range(self.num_cameras):
                sensor_configs.append(CameraConfig(uid=f"base_camera_{i}",
                                                pose=pose,
                                                width=self.camera_width,
                                                height=self.camera_height,
                                                fov=np.pi / 2))
        return sensor_configs

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)
