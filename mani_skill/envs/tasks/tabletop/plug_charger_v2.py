import numpy as np

from mani_skill.envs.tasks.tabletop.plug_charger import PlugChargerEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env


@register_env("PlugCharger-v2", max_episode_steps=200)
class PlugChargerV2Env(PlugChargerEnv):
    """
    **Task Description:**
    Nearly exacty copy of PlugChargerEnv, but with 3 cameras instead of 1.
    """
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        assert "camera_width" in kwargs, "camera_width must be provided"
        assert "camera_height" in kwargs, "camera_height must be provided"
        self._camera_width = kwargs.pop("camera_width")
        self._camera_height = kwargs.pop("camera_height")
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose_center = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        pose_left = sapien_utils.look_at(eye=[0.0, -0.3, 0.6], target=[-0.1, 0, 0.1])
        pose_right = sapien_utils.look_at(eye=[0.0, 0.3, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("camera_center", pose_center, self._camera_width, self._camera_height, np.pi / 2, 0.01, 100),
            CameraConfig("camera_left", pose_left, self._camera_width, self._camera_height, np.pi / 2, 0.01, 100),
            CameraConfig("camera_right", pose_right, self._camera_width, self._camera_height, np.pi / 2, 0.01, 100),
        ]