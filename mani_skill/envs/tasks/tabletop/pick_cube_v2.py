import numpy as np

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env

REALSENSE_DEPTH_FOV_VERTICAL_RAD = 58.0 * np.pi / 180
REALSENSE_DEPTH_FOV_HORIZONTAL_RAD = 87.0 * np.pi / 180


@register_env("PickCube-v2", max_episode_steps=100)
class PickCubeV2Env(PickCubeEnv):
    """
    **Task Description:**
    Nearly exacty copy of PickCubeEnv, but with 3 cameras instead of 1.
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
        SHADER = "default"
        return [
            CameraConfig(
                uid="camera_center",
                pose=pose_center,
                width=self._camera_width,
                height=self._camera_height,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                shader_pack=SHADER,
            ),
            CameraConfig(
                uid="camera_left",
                pose=pose_left,
                width=self._camera_width,
                height=self._camera_height,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                shader_pack=SHADER,
            ),
            CameraConfig(
                uid="camera_right",
                pose=pose_right,
                width=self._camera_width,
                height=self._camera_height,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                shader_pack=SHADER,
            ),
        ]