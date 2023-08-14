from real_robot.utils.registration import register_env
from mani_skill2.envs.pick_and_place.place_cube_in_bowl import (
    PlaceCubeInBowlEnv
)
from .base_env import XArmBaseEnv


@register_env("PlaceCubeInBowlRealXArm-v8", max_episode_steps=50,
              reward_mode="dense_v2",
              robot="xarm7_d435", image_obs_mode="hand_front",
              no_static_checks=True, success_needs_ungrasp=True,
              success_cube_above_only=True, goal_height_delta=0.08)
class PlaceCubeInBowlRealEnv(XArmBaseEnv, PlaceCubeInBowlEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
