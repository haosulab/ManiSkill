from real_robot.utils.registration import register_env
from mani_skill2.envs.pick_and_place.pick_cube import (
    PickCubeEnv_v1
)
from .base_env import XArmBaseEnv


@register_env("PickCubeRealXArm", max_episode_steps=50,)
class PickCubeRealEnv(XArmBaseEnv, PickCubeEnv_v1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
