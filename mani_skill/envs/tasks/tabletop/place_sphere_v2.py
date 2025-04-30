import numpy as np
from typing import Union

import torch
import sapien

from mani_skill.utils.structs import Link, Actor
import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.tasks.tabletop.place_sphere import PlaceSphereEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
from mani_skill.envs.distraction_set import DistractionSet

from mani_skill.envs.tasks.tabletop.get_camera_config import get_camera_configs, get_human_render_camera_config

DEFAULT_GOAL_THRESH_MARGIN = 0.05

@register_env("PlaceSphere-v2", max_episode_steps=100)
class PlaceSphereV2Env(PlaceSphereEnv):
    """
    **Task Description:**
    Nearly exacty copy of PickCubeEnv, but with the following change:
        1. 3 cameras instead of 1
        2. Cameras have a higher resolution
        3. Target position is fixed to (0.05, 0.05, 0.25)
        4. Goal_thresh is the cube half size plus a configurable margin
    """
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, goal_thresh_margin=DEFAULT_GOAL_THRESH_MARGIN, **kwargs):
        assert "camera_width" in kwargs, "camera_width must be provided"
        assert "camera_height" in kwargs, "camera_height must be provided"
        assert "distraction_set" in kwargs, "distraction_set must be provided"
        self._camera_width = kwargs.pop("camera_width")
        self._camera_height = kwargs.pop("camera_height")
        self._distraction_set: Union[DistractionSet, dict] = kwargs.pop("distraction_set")
        self._goal_thresh_margin = goal_thresh_margin
        # In this situation, the DistractionSet has serialized as a dict so we now need to deserialize it.
        if isinstance(self._distraction_set, dict):
            self._distraction_set = DistractionSet(**self._distraction_set)
        self._human_render_shader = kwargs.pop("human_render_shader", None)
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)
        print(" --> Created PlaceSphereV2Env")

    @property
    def _default_human_render_camera_configs(self):
        return get_human_render_camera_config(eye=[0.5, 0.5, 0.25], target=[0.0, 0.0, 0.1], shader=self._human_render_shader)

    @property
    def _default_sensor_configs(self):
        target=[-0.1, 0, 0.1]
        eye_xy = 0.3
        eye_z = 0.6
        cfgs = get_camera_configs(eye_xy, eye_z, target, self._camera_width, self._camera_height)
        cfgs_adjusted = self._distraction_set.update_camera_configs(cfgs)
        return cfgs_adjusted