import numpy as np

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env

from mani_skill.envs.tasks.tabletop.get_camera_config import get_camera_configs, get_human_render_camera_config

@register_env("StackCube-v2", max_episode_steps=50)
class StackCubeV2Env(StackCubeEnv):
    """
    Derived from StackCubeEnv, but with 3 cameras instead of 1. The dimensions of the cameras can be set as well.
    """
    def __init__(
        self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs
    ):
        assert "camera_width" in kwargs, "camera_width must be provided"
        assert "camera_height" in kwargs, "camera_height must be provided"
        self._camera_width = kwargs.pop("camera_width")
        self._camera_height = kwargs.pop("camera_height")
        self._distraction_set = kwargs.pop("distraction_set")
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        return get_human_render_camera_config(eye=[0.3, 0, 0.4], target=[-0.1, 0, 0.1])

    @property
    def _default_sensor_configs(self):
        target = [0, 0, 0.0]
        eye_xy = 0.3
        eye_z = 0.4
        cfgs = get_camera_configs(eye_xy, eye_z, target, self._camera_width, self._camera_height)
        return cfgs