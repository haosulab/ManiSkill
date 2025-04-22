import numpy as np
import torch
from transforms3d.euler import euler2quat
from mani_skill.utils.structs import Pose

from mani_skill.envs.tasks.tabletop.lift_peg_upright import LiftPegUprightEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.tabletop.get_camera_config import get_camera_configs, get_human_render_camera_config
from mani_skill.envs.distraction_set import DistractionSet

@register_env("LiftPegUpright-v2", max_episode_steps=50)
class LiftPegUprightV2Env(LiftPegUprightEnv):
    """
    **Task Description:**
    Nearly exacty copy of PullCubeToolEnv, but with 3 cameras instead of 1. Also the peg spawn bounds have -x by 10cm to prevent the joint limits from being reached, which creates weird / slow / bobbing motion plans otherwise (~30% of the time).

    Appropriate pointcloud bounds:
    x: [-0.3, 0.3] 
    y: [-0.3, 0.3] 
    z: [0.0, 0.5]
    """
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        assert "camera_width" in kwargs, "camera_width must be provided"
        assert "camera_height" in kwargs, "camera_height must be provided"
        self._camera_width = kwargs.pop("camera_width")
        self._camera_height = kwargs.pop("camera_height")
        self._distraction_set = kwargs.pop("distraction_set")
        if isinstance(self._distraction_set, dict):
            self._distraction_set = DistractionSet(**self._distraction_set)

        self.peg_spawn_bounds_x = [-0.2, 0]
        self.peg_spawn_bounds_y = [-0.1, 0.1]
        
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            x_range = self.peg_spawn_bounds_x[1] - self.peg_spawn_bounds_x[0]
            y_range = self.peg_spawn_bounds_y[1] - self.peg_spawn_bounds_y[0]
            xyz[:, 0] = torch.rand((b,)) * x_range + self.peg_spawn_bounds_x[0]
            xyz[:, 1] = torch.rand((b,)) * y_range + self.peg_spawn_bounds_y[0]
            xyz[:, 2] = self.peg_half_width
            q = euler2quat(np.pi / 2, 0, 0)

            
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.peg.set_pose(obj_pose)


    @property
    def _default_human_render_camera_configs(self):
        return get_human_render_camera_config(eye=[0.4, 0.5, 0.4], target=[0.0, 0.0, 0.1])

    @property
    def _default_sensor_configs(self):
        target = [0.0, 0, 0.1]
        eye_xy = 0.5
        eye_z = 0.6
        return get_camera_configs(eye_xy, eye_z, target, self._camera_width, self._camera_height)