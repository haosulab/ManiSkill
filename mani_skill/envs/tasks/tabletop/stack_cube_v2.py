import numpy as np

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env

DEFAULT_CAMERA_WIDTH = 128
DEFAULT_CAMERA_HEIGHT = 128

@register_env("StackCube-v2", max_episode_steps=50)
class StackCubeV2Env(StackCubeEnv):
    """
    Derived from StackCubeEnv, but with 3 cameras instead of 1. The dimensions of the cameras can be set as well.
    """
    def __init__(
        self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs
    ):
        self._camera_width = kwargs.pop("camera_width", DEFAULT_CAMERA_WIDTH)
        self._camera_height = kwargs.pop("camera_height", DEFAULT_CAMERA_HEIGHT)
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        target_position = [-0.1, 0, 0.1]
        pose_center = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=target_position)
        pose_left = sapien_utils.look_at(eye=[0.0, -0.3, 0.6], target=target_position)
        pose_right = sapien_utils.look_at(eye=[0.0, 0.3, 0.6], target=target_position)
        return [
            CameraConfig("camera_center", pose_center, self._camera_width, self._camera_height, np.pi / 2, 0.01, 100),
            CameraConfig("camera_left", pose_left, self._camera_width, self._camera_height, np.pi / 2, 0.01, 100),
            CameraConfig("camera_right", pose_right, self._camera_width, self._camera_height, np.pi / 2, 0.01, 100),
        ]

