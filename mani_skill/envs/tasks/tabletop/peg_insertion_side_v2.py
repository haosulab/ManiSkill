import numpy as np

from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.tabletop.get_camera_config import get_camera_configs, get_human_render_camera_config

@register_env("PegInsertionSide-v2", max_episode_steps=50)
class PegInsertionSideV2Env(PegInsertionSideEnv):
    """
    **Task Description:**
    Nearly exacty copy of PegInsertionSideEnv, but with 3 cameras instead of 1.
    """
    def __init__(self, *args, **kwargs):
        assert "camera_width" in kwargs, "camera_width must be provided"
        assert "camera_height" in kwargs, "camera_height must be provided"
        self._camera_width = kwargs.pop("camera_width")
        self._camera_height = kwargs.pop("camera_height")
        self._distraction_set = kwargs.pop("distraction_set")
        super().__init__(*args, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        return get_human_render_camera_config(eye=[0.45, -0.45, 0.7], target=[0.05, -0.1, 0.4])


    @property
    def _default_sensor_configs(self):
        target = [0, 0, 0.1]
        eye_xy = 0.3
        eye_z = 0.4
        return get_camera_configs(eye_xy, eye_z, target, self._camera_width, self._camera_height)