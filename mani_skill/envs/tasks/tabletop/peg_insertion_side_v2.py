import numpy as np

from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env


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
        super().__init__(*args, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        """ Configures the human render camera.
        """
        pose = sapien_utils.look_at([0.45, -0.45, 0.7], [0.05, -0.1, 0.4])
        SHADER = "default"
        return CameraConfig("render_camera", pose=pose, width=1264, height=1264, fov=np.pi / 3, near=0.01, far=100, shader_pack=SHADER)


    @property
    def _default_sensor_configs(self):
        target = [0, 0, 0.1]
        pose_center = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=target)
        pose_left = sapien_utils.look_at(eye=[0.0, -0.3, 0.6], target=target)
        pose_right = sapien_utils.look_at(eye=[0.0, 0.3, 0.6], target=target)
        return [
            CameraConfig("camera_center", pose_center, self._camera_width, self._camera_height, np.pi / 2, 0.01, 100),
            CameraConfig("camera_left", pose_left, self._camera_width, self._camera_height, np.pi / 2, 0.01, 100),
            CameraConfig("camera_right", pose_right, self._camera_width, self._camera_height, np.pi / 2, 0.01, 100),
        ]