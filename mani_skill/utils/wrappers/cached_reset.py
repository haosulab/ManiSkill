from dataclasses import asdict, dataclass
from typing import List, Optional, Union

import dacite
import gymnasium as gym
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, tree
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
        config: A dictionary or a `CachedResetsConfig` object that contains the configuration for the cached resets.
    """

    def __init__(
        self,
        env: gym.Env,
        reset_to_env_states: Optional[dict] = None,
        config: Union[CachedResetsConfig, dict] = CachedResetsConfig(),
    ):
        super().__init__(env)
        self.num_envs = self.base_env.num_envs
        if isinstance(config, CachedResetsConfig):
            config = config.dict()
        self.cached_resets_config = dacite.from_dict(
            data_class=CachedResetsConfig,
            data=config,
            config=dacite.Config(strict=True),
        )
        cached_data_device = self.cached_resets_config.device
        if cached_data_device is None:
            cached_data_device = self.base_env.device
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
                                self.cached_resets_config.num_resets
                                - self._num_cached_resets,
                                self.num_envs,
                            ),
                            device=self.base_env.device,
                        )
                    ),
                )
                state = self.env.get_wrapper_attr("get_state_dict")()
                if (
                    self.cached_resets_config.num_resets - self._num_cached_resets
                    < self.num_envs
                ):
                    obs = tree.slice(
                        obs,
                        slice(
                            0,
                            self.cached_resets_config.num_resets
                            - self._num_cached_resets,
                        ),
                    )
                    state = tree.slice(
                        state,
                        slice(
                            0,
                            self.cached_resets_config.num_resets
                            - self._num_cached_resets,
                        ),
                    )
                self._cached_resets_obs_buffer.append(
                    common.to_tensor(obs, device=self.cached_resets_config.device)
                )
                self._cached_resets_env_states.append(
                    common.to_tensor(state, device=self.cached_resets_config.device)
                )
                self._num_cached_resets += self.num_envs
            self._cached_resets_env_states = tree.cat(self._cached_resets_env_states)
            self._cached_resets_obs_buffer = tree.cat(self._cached_resets_obs_buffer)

        self._cached_resets_env_states = common.to_tensor(
            self._cached_resets_env_states, device=cached_data_device
        )
        if self._cached_resets_obs_buffer is not None:
            self._cached_resets_obs_buffer = common.to_tensor(
                self._cached_resets_obs_buffer, device=cached_data_device
            )

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def reset(
        self,
        *args,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
        **kwargs
    ):
        env_idx = None
        if options is None:
            options = dict()
        if "env_idx" in options:
            env_idx = options["env_idx"]
        if self._cached_resets_env_states is not None:
            sampled_ids = torch.randint(
                0,
                self._num_cached_resets,
                size=(len(env_idx) if env_idx is not None else self.num_envs,),
                device=self.base_env.device,
            )
            options["reset_to_env_states"] = dict(
                env_states=tree.slice(self._cached_resets_env_states, sampled_ids),
            )
            if self._cached_resets_obs_buffer is not None:
                options["reset_to_env_states"]["obs"] = tree.slice(
                    self._cached_resets_obs_buffer, sampled_ids
                )
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info
