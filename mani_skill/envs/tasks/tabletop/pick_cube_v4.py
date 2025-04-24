import numpy as np
from typing import Union

import torch
import sapien

from mani_skill.envs.tasks.tabletop.pick_cube_v2 import PickCubeV2Env
from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.tabletop.get_camera_config import get_camera_configs


@register_env("PickCube-v4", max_episode_steps=100)
class PickCubeV4Env(PickCubeV2Env):

    # The calculation:
    # cube_half_size = 0.02
    # goal-threshold_radius = cube_half_size + goal_thresh_margin -> 0.06, diameter = 0.12
    goal_thresh_margin = 0.04

    """
    **Task Description:**
    Copy of PickCubeEnvV2, but the goal-threshold is larger.
    """
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, goal_thresh_margin=self.goal_thresh_margin, **kwargs)

    @property
    def _default_sensor_configs(self):
        eye_xy = 0.3
        eye_z = 0.4
        target = [0.0, 0, 0.15]
        cfgs = get_camera_configs(eye_xy, eye_z, target, self._camera_width, self._camera_height)
        cfgs_adjusted = self._distraction_set.update_camera_configs(cfgs)
        return cfgs_adjusted