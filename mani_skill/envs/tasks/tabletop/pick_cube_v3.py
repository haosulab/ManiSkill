import numpy as np
from typing import Union

import torch
import sapien

from mani_skill.envs.tasks.tabletop.pick_cube_v2 import PickCubeV2Env
from mani_skill.utils.registration import register_env
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig


REALSENSE_DEPTH_FOV_VERTICAL_RAD = 58.0 * np.pi / 180
REALSENSE_DEPTH_FOV_HORIZONTAL_RAD = 87.0 * np.pi / 180


@register_env("PickCube-v3", max_episode_steps=100)
class PickCubeV3Env(PickCubeV2Env):

    goal_thresh_margin = 0.01 # used during training

    """
    **Task Description:**
    Copy of PickCubeEnvV2, but the cameras are closer to the cube.
    """
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, goal_thresh_margin=self.goal_thresh_margin, **kwargs)
        print(" --> Created PickCubeV3Env")

    @property
    def _default_sensor_configs(self):
        pose_center = sapien_utils.look_at(eye=[0.3, 0, 0.4], target=[0.0, 0, 0.15])
        pose_left = sapien_utils.look_at(eye=[0.0, -0.3, 0.4], target=[0.0, 0, 0.15])
        pose_right = sapien_utils.look_at(eye=[0.0, 0.3, 0.4], target=[0.0, 0, 0.15])
        SHADER = "default"
        cfgs = [
            CameraConfig(
                uid="camera_center",
                pose=pose_center,
                width=self._camera_width,
                height=self._camera_height,
                fov=REALSENSE_DEPTH_FOV_VERTICAL_RAD,
                near=0.01,
                far=100,
                shader_pack=SHADER,
            ),
            CameraConfig(
                uid="camera_left",
                pose=pose_left,
                width=self._camera_width,
                height=self._camera_height,
                fov=REALSENSE_DEPTH_FOV_VERTICAL_RAD,
                near=0.01,
                far=100,
                shader_pack=SHADER,
            ),
            CameraConfig(
                uid="camera_right",
                pose=pose_right,
                width=self._camera_width,
                height=self._camera_height,
                fov=REALSENSE_DEPTH_FOV_VERTICAL_RAD,
                near=0.01,
                far=100,
                shader_pack=SHADER,
            ),
        ]
        cfgs_adjusted = self._distraction_set.update_camera_configs(cfgs)
        return cfgs_adjusted