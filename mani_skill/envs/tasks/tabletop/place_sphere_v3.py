import numpy as np
from typing import Union

import torch
import sapien

from mani_skill.envs.tasks.tabletop.place_sphere_v2 import PlaceSphereV2Env
from mani_skill.utils.registration import register_env
from mani_skill.sensors.camera import CameraConfig
from mani_skill.envs.tasks.tabletop.get_camera_config import get_camera_configs


@register_env("PlaceSphere-v3", max_episode_steps=100)
class PlaceSphereV3Env(PlaceSphereV2Env):

    # Originally:
    # radius = 0.02  # radius of the sphere
    # inner_side_half_len = 0.02  # side length of the bin's inner square
    # short_side_half_size = 0.0025  # length of the shortest edge of the block

    # Modified:
    radius = 0.03  # radius of the sphere
    inner_side_half_len = 0.03  # side length of the bin's inner square
    short_side_half_size = 0.005  # length of the shortest edge of the block
    block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size + inner_side_half_len,
    ]  # The bottom block of the bin, which is larger: The list represents the half length of the block along the [x, y, z] axis respectively.
    edge_block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size,
    ]  # The edge block of the bin, which is smaller. The representations are similar to the above one


    """
    **Task Description:**
    Copy of PickCubeEnvV2, but the cameras are closer to the cube.
    """
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)
