import gymnasium as gym


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
