import numpy as np
from typing import Union

import torch
import sapien

from mani_skill.envs.tasks.tabletop.pick_cube_v2 import PickCubeV2Env
from mani_skill.utils.registration import register_env
from mani_skill.sensors.camera import CameraConfig
from mani_skill.envs.tasks.tabletop.get_camera_config import get_camera_configs


@register_env("PickCube-v3", max_episode_steps=100)
class PickCubeV3Env(PickCubeV2Env):

    goal_thresh_margin = 0.01

    """
    **Task Description:**
    Copy of PickCubeEnvV2, but the cameras are closer to the cube.
    """
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, goal_thresh_margin=self.goal_thresh_margin, **kwargs)
        print(" --> Created PickCubeV3Env")

    @property
    def _default_sensor_configs(self):
        print("  PickCubeV3Env: _default_sensor_configs()")
        target=[0.0, 0, 0.15]
        xy_offset = 0.3
        z_offset = 0.4
        cfgs = get_camera_configs(xy_offset, z_offset, target, self._camera_width, self._camera_height)
        cfgs_adjusted = self._distraction_set.update_camera_configs(cfgs)
        return cfgs_adjusted