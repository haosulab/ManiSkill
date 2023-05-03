import gymnasium as gym


class ContinuousTaskWrapper(gym.Wrapper):
    """
    Defines a continuous, infinite horizon, task where terminated is always False
    unless a timelimit is reached.
    """

    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)

    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        return ob, rew, False, truncated, info


class SuccessInfoWrapper(gym.Wrapper):
    """
    A simple wrapper that adds a is_success key which SB3 tracks
    """

    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        info["is_success"] = info["success"]
        return ob, rew, terminated, truncated, info
