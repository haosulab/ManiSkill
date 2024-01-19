from collections import OrderedDict

import numpy as np
import sapien
import sapien.physx as physx

from mani_skill2.utils.registration import register_env
from mani_skill2.utils.structs.pose import vectorize_pose

from .base_env import SceneManipulationEnv


@register_env("PickObjectScene-v0", max_episode_steps=200)
class PickObjectSceneEnv(SceneManipulationEnv):
    """
    Args:

    """

    def __init__(self, *args, **kwargs):
        self.box_length = 0.1
        super().__init__(*args, **kwargs)

    def _load_actors(self):
        super()._load_actors()

    def reconfigure(self):
        super().reconfigure()
        self.init_state = self.get_state()

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.agent.tcp.pose),
            obj_pose=vectorize_pose(self.goal_obj.pose),
        )
        return obs

    def evaluate(self, **kwargs):
        return dict(success=self.agent.is_grasping(self.goal_obj))

    def compute_dense_reward(self, info, **kwargs):
        return 1 - np.tanh(np.linalg.norm(self.goal_obj.pose.p - self.agent.tcp.pose.p))

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 1

    def _initialize_actors(self):
        self.set_state(self.init_state)

    def _initialize_task(self):
        # pick a random goal object to pick up
        self.goal_obj: sapien.Entity = self._episode_rng.choice(
            self.scene_builder.movable_objects
        )
        print(f"Target Object: {self.goal_obj.name}")

    def reset(self, seed=None, options=None):
        # sample a scene
        return super().reset(seed, options)
