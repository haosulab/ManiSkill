import gymnasium as gym

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.common import flatten_state_dict


class FlattenObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the observations into a single vector
    """

    def __init__(self, env) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.base_env._update_obs_space(self.observation(self.base_env._init_raw_obs))

    def observation(self, observation):
        return flatten_state_dict(observation)
