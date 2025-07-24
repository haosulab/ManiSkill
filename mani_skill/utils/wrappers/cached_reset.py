import gymnasium as gym
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, tree
import torch

class CachedResetWrapper(gym.Wrapper):
    """
    Cached reset wrapper for ManiSkill3 environments.

    Args:
        env: The environment to wrap.
        reset_to_env_states: A dictionary with keys "env_states" and optionally "obs". "env_states" is a dictionary of environment states to reset to. 
            "obs" contains the corresponding observations generated at those env states. If reset_to_env_states is not provided, the wrapper will sample reset states
            from the environment using the given seed.
        cached_resets_config: 
    """
    def __init__(self, env: gym.Env, reset_to_env_states: dict = None, cached_resets_config: dict = None):
        super().__init__(env)
        self.reset_to_env_states = reset_to_env_states
        self.num_envs = self.base_env.num_envs
        # reset caching
        # self.cached_resets = cached_resets
        # self.cached_resets_config = cached_resets_config
        if self.reset_to_env_states is None:
            if self.cached_resets_config.num_resets is None:
                self.cached_resets_config.num_resets = 16384
            count = 0
            self._cached_resets_env_states = []
            self._cached_resets_obs_buffer = []
            while count < self.cached_resets_config.num_resets:
                obs, _ = self._env.reset(
                    seed=self.cached_resets_config.seed,
                    options=dict(
                        env_idx=torch.arange(
                            0,
                            min(
                                self.cached_resets_config.num_resets - count,
                                self.num_envs,
                            ),
                            device=self.device,
                        )
                    ),
                )
                state = self._env.get_state()
                if self.cached_resets_config.num_resets - count < self.num_envs:
                    obs = tree.slice(
                        obs, slice(0, self.cached_resets_config.num_resets - count)
                    )
                    state = tree.slice(
                        state, slice(0, self.cached_resets_config.num_resets - count)
                    )
                self._cached_resets_obs_buffer.append(
                    common.to_tensor(obs, device=self.cached_resets_config.device)
                )
                self._cached_resets_env_states.append(
                    common.to_tensor(state, device=self.cached_resets_config.device)
                )
                count += len(state)
            self._cached_resets_env_states = tree.cat(self._cached_resets_env_states)
            self._cached_resets_obs_buffer = tree.cat(self._cached_resets_obs_buffer)
            self._last_obs = tree.slice(
                self._cached_resets_obs_buffer,
                torch.arange(0, self.num_envs, device=self.device),
            )

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def reset(self, seed=None, options=None):
        if "env_idx" in options:
            env_idx = options["env_idx"]
        # sample reset states
        sampled_ids = torch.randint(
            0,
            len(self.reset_to_env_states),
            size=(len(env_idx),),
            device=self.base_env.device,
        )
        
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info