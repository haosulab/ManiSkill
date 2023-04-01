import numpy as np
from gymnasium import spaces


class BasePolicy:
    def __init__(
        self, env_id: str, observation_space: spaces.Space, action_space: spaces.Space
    ) -> None:
        self.env_id = env_id
        self.observation_space = observation_space
        self.action_space = action_space
        # NOTE(jigu): Do not assume that gym.make(env_id) works during evaluation

    def reset(self, observations):
        """Called at the beginning of an episode."""
        pass

    def act(self, observations) -> np.ndarray:
        """Act based on the observations."""
        raise NotImplementedError

    @classmethod
    def get_obs_mode(cls, env_id: str) -> str:
        """Get the observation mode for the policy. Define the observation space."""
        raise NotImplementedError

    @classmethod
    def get_control_mode(cls, env_id) -> str:
        """Get the control mode for the policy. Define the action space."""
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    def act(self, observations):
        return self.action_space.sample()

    @classmethod
    def get_obs_mode(cls, env_id: str) -> str:
        return "rgbd"

    @classmethod
    def get_control_mode(cls, env_id: str) -> str:
        return None  # use default one
