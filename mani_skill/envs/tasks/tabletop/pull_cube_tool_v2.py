import numpy as np

from mani_skill.envs.tasks.tabletop.pull_cube_tool import PullCubeToolEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.tabletop.get_camera_config import get_camera_configs

@register_env("PullCubeTool-v2", max_episode_steps=50)
class PullCubeToolV2Env(PullCubeToolEnv):
    """
    **Task Description:**
    Nearly exacty copy of PullCubeToolEnv, but with 3 cameras instead of 1.
    """
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        assert "camera_width" in kwargs, "camera_width must be provided"
        assert "camera_height" in kwargs, "camera_height must be provided"
        self._camera_width = kwargs.pop("camera_width")
        self._camera_height = kwargs.pop("camera_height")
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        """ Configures the human render camera.
        """
        pose = sapien_utils.look_at([0.5, 0.6, 0.5], [0.0, 0.0, 0.35])
        SHADER = "default"
        return CameraConfig("render_camera", pose=pose, width=1264, height=1264, fov=np.pi / 3, near=0.01, far=100, shader_pack=SHADER)


    @property
    def _default_sensor_configs(self):
        target = [0, 0, 0.0]
        eye_xy = 0.3
        eye_z = 0.4
        cfgs = get_camera_configs(eye_xy, eye_z, target, self._camera_width, self._camera_height)
        return cfgs