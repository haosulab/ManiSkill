import dacite
import gymnasium as gym
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, tree
import torch
from typing import Optional, Union
from dataclasses import asdict, dataclass
from mani_skill.utils.structs.types import Device

@dataclass
class CachedResetsConfig:
    num_resets: Optional[int] = None
    """The number of reset states to cache. If none it will cache `num_envs` number of reset states."""
    device: Optional[Device] = None
    """The device to cache the reset states on. If none it will use the base environment's device."""
    seed: Optional[int] = None
    """The seed to use for generating the cached reset states."""

    def dict(self):
        return {k: v for k, v in asdict(self).items()}

class CachedResetWrapper(gym.Wrapper):
    """
    Cached reset wrapper for ManiSkill3 environments. Caching resets allows you to skip slower parts of the reset function call and boost environment FPS as a result.

    Args:
        env: The environment to wrap.
        reset_to_env_states: A dictionary with keys "env_states" and optionally "obs". "env_states" is a dictionary of environment states to reset to. 
            "obs" contains the corresponding observations generated at those env states. If reset_to_env_states is not provided, the wrapper will sample reset states
            from the environment using the given seed.
        cached_resets_config: 
    """
    def __init__(self, env: gym.Env, reset_to_env_states: dict = None, cached_resets_config: Union[CachedResetsConfig, dict] = CachedResetsConfig()):
        super().__init__(env)
        self.num_envs = self.base_env.num_envs
        if isinstance(cached_resets_config, CachedResetsConfig):
            cached_resets_config = cached_resets_config.dict()
        self.cached_resets_config = dacite.from_dict(data_class=CachedResetsConfig, data=cached_resets_config, config=dacite.Config(strict=True))
        
        self._num_cached_resets = 0
        if reset_to_env_states is not None:
            self._cached_resets_env_states = reset_to_env_states["env_states"]
            self._cached_resets_obs_buffer = reset_to_env_states.get("obs", None)
            self._num_cached_resets = len(self._cached_resets_env_states)
        else:
            if self.cached_resets_config.num_resets is None:
                self.cached_resets_config.num_resets = 16384
            self._cached_resets_env_states = []
            self._cached_resets_obs_buffer = []
            while self._num_cached_resets < self.cached_resets_config.num_resets:
                obs, _ = self.env.reset(
                    seed=self.cached_resets_config.seed,
                    options=dict(
                        env_idx=torch.arange(
                            0,
                            min(
                                self.cached_resets_config.num_resets - self._num_cached_resets,
                                self.num_envs,
                            ),
                            device=self.device,
                        )
                    ),
                )
                state = self.env.get_wrapper_attr("get_state")()
                if self.cached_resets_config.num_resets - self._num_cached_resets < self.num_envs:
                    obs = tree.slice(
                        obs, slice(0, self.cached_resets_config.num_resets - self._num_cached_resets)
                    )
                    state = tree.slice(
                        state, slice(0, self.cached_resets_config.num_resets - self._num_cached_resets)
                    )
                self._cached_resets_obs_buffer.append(
                    common.to_tensor(obs, device=self.cached_resets_config.device)
                )
                self._cached_resets_env_states.append(
                    common.to_tensor(state, device=self.cached_resets_config.device)
                )
                self._num_cached_resets += len(state)
            self._cached_resets_env_states = tree.cat(self._cached_resets_env_states)
            self._cached_resets_obs_buffer = tree.cat(self._cached_resets_obs_buffer)

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def reset(self, seed=None, options=None):
        env_idx = None
        if "env_idx" in options:
            env_idx = options["env_idx"]
        if self._cached_resets_env_states is not None:
            # sample reset states
            sampled_ids = torch.randint(
                0,
                self._num_cached_resets,
                size=(len(env_idx) if env_idx is not None else self.num_envs, ),
                device=self.base_env.device,
            )
            options["reset_to_env_states"] = dict(
                env_states=tree.slice(self._cached_resets_env_states, sampled_ids),
                obs=tree.slice(self._cached_resets_obs_buffer, sampled_ids)
            )
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info