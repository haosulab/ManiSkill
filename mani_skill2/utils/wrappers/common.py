import gym
from gym import spaces

from ..common import (
    clip_and_scale_action,
    inv_clip_and_scale_action,
    normalize_action_space,
)


class NormalizeBoxActionWrapper(gym.ActionWrapper):
    """Normalize box action space to [-1, 1]."""

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.env.action_space, spaces.Box), self.env.action_space
        self.action_space = normalize_action_space(self.env.action_space)

    def action(self, action):
        return clip_and_scale_action(
            action, self.env.action_space.low, self.env.action_space.high
        )

    def reverse_action(self, action):
        return inv_clip_and_scale_action(
            action, self.env.action_space.low, self.env.action_space.high
        )


class ResetSeedWrapper(gym.Wrapper):
    """Reset env with a fixed seed.
    It assumes that observation and action spaces do not depend on seed.
    """

    def __init__(self, env, reset_seed: int) -> None:
        super().__init__(env)
        self.reset_seed = reset_seed

    def reset(self, **kwargs):
        kwargs.setdefault("seed", self.reset_seed)
        return super().reset(**kwargs)
