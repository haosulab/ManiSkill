from __future__ import annotations

import json
import sys
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Dict, List, Optional, Type

import gymnasium as gym
import torch
from gymnasium.envs.registration import EnvSpec as GymEnvSpec
from gymnasium.envs.registration import WrapperSpec

from mani_skill import logger
from mani_skill.utils import assets, download_asset
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

if TYPE_CHECKING:
    from mani_skill.envs.sapien_env import BaseEnv


class EnvSpec:
    def __init__(
        self,
        uid: str,
        cls: Type[BaseEnv],
        max_episode_steps=None,
        asset_download_ids: Optional[List[str]] = [],
        default_kwargs: dict = None,
    ):
        """A specification for a ManiSkill environment."""
        self.uid = uid
        self.cls = cls
        self.max_episode_steps = max_episode_steps
        self.asset_download_ids = asset_download_ids
        self.default_kwargs = {} if default_kwargs is None else default_kwargs

    def make(self, **kwargs):
        _kwargs = self.default_kwargs.copy()
        _kwargs.update(kwargs)

        # check if all assets necessary are downloaded
        assets_to_download = []
        for asset_id in self.asset_download_ids or []:
            is_data_group = asset_id in assets.DATA_GROUPS
            if is_data_group:
                found_data_group_assets = True
                for (
                    data_source_id
                ) in assets.expand_data_group_into_individual_data_source_ids(asset_id):
                    if not assets.is_data_source_downloaded(data_source_id):
                        assets_to_download.append(data_source_id)
                        found_data_group_assets = False
                if not found_data_group_assets:
                    print(
                        f"Environment {self.uid} requires a set of assets in group {asset_id}. At least 1 of those assets could not be found"
                    )
            else:
                if not assets.is_data_source_downloaded(asset_id):
                    assets_to_download.append(asset_id)
                    data_source = assets.DATA_SOURCES[asset_id]
                    print(
                        f"Could not find asset {asset_id} at {data_source.output_dir / data_source.target_path}"
                    )
        if len(assets_to_download) > 0:
            if len(assets_to_download) <= 5:
                asset_download_msg = ", ".join(assets_to_download)
            else:
                asset_download_msg = f"{assets_to_download[:5]} (and {len(assets_to_download) - 10} more)"
            response = download_asset.prompt_yes_no(
                f"Environment {self.uid} requires asset(s) {asset_download_msg} which could not be found. Would you like to download them now?"
            )
            if response:
                for asset_id in assets_to_download:
                    download_asset.download(assets.DATA_SOURCES[asset_id])
            else:
                print("Exiting as assets are not found or downloaded")
                exit()
        return self.cls(**_kwargs)

    @property
    def gym_spec(self):
        """Return a gym EnvSpec for this env"""
        entry_point = self.cls.__module__ + ":" + self.cls.__name__
        return GymEnvSpec(
            self.uid,
            entry_point,
            max_episode_steps=self.max_episode_steps,
            kwargs=self.default_kwargs,
        )


REGISTERED_ENVS: Dict[str, EnvSpec] = {}


def register(
    name: str,
    cls: Type[BaseEnv],
    max_episode_steps=None,
    asset_download_ids: List[str] = [],
    default_kwargs: dict = None,
):
    """Register a ManiSkill environment."""

    # hacky way to avoid circular import errors when users inherit a task in ManiSkill and try to register it themselves
    from mani_skill.envs.sapien_env import BaseEnv

    if name in REGISTERED_ENVS:
        logger.warn(f"Env {name} already registered")
    if not issubclass(cls, BaseEnv):
        raise TypeError(f"Env {name} must inherit from BaseEnv")

    for asset_id in asset_download_ids:
        if asset_id not in assets.DATA_SOURCES and asset_id not in assets.DATA_GROUPS:
            raise KeyError(f"Asset {asset_id} not found in data sources or groups")

    REGISTERED_ENVS[name] = EnvSpec(
        name,
        cls,
        max_episode_steps=max_episode_steps,
        asset_download_ids=asset_download_ids,
        default_kwargs=default_kwargs,
    )
    assets.DATA_GROUPS[name] = asset_download_ids


class TimeLimitWrapper(gym.Wrapper):
    """like the standard gymnasium timelimit wrapper but fixes truncated variable to be a torch tensor and batched"""

    def __init__(self, env: gym.Env, max_episode_steps: int):
        super().__init__(env)
        prev_frame_locals = sys._getframe(1).f_locals
        frame = sys._getframe(1)
        # check for user supplied max_episode_steps during gym.make calls
        if frame.f_code.co_name == "make" and "max_episode_steps" in prev_frame_locals:
            if prev_frame_locals["max_episode_steps"] is not None:
                max_episode_steps = prev_frame_locals["max_episode_steps"]
            # do some wrapper surgery to remove the previous timelimit wrapper
            # with gymnasium 0.29.1, this will remove the timelimit wrapper and nothing else.
            curr_env = env
            found_env = False
            while curr_env is not None:
                if isinstance(curr_env, gym.wrappers.TimeLimit):
                    self.env = curr_env.env
                    found_env = True
                    break
                if hasattr(curr_env, "env"):
                    curr_env = curr_env.env
                else:
                    break
            if not found_env:
                self.env = env

        self._max_episode_steps = max_episode_steps

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self._max_episode_steps is not None:
            truncated = self.base_env.elapsed_steps >= self._max_episode_steps
        else:
            truncated = torch.zeros(
                (self.base_env.num_envs,), dtype=torch.bool, device=self.base_env.device
            )
        return observation, reward, terminated, truncated, info


def make(env_id, **kwargs):
    """Instantiate a ManiSkill environment.

    Args:
        env_id (str): Environment ID.
        as_gym (bool, optional): Add TimeLimit wrapper as gym.
        **kwargs: Keyword arguments to pass to the environment.
    """
    if env_id not in REGISTERED_ENVS:
        raise KeyError("Env {} not found in registry".format(env_id))
    env_spec = REGISTERED_ENVS[env_id]
    env = env_spec.make(**kwargs)
    return env


def make_vec(env_id, **kwargs):
    env = gym.make(env_id, **kwargs)
    env = ManiSkillVectorEnv(env)
    return env


def register_env(
    uid: str,
    max_episode_steps=None,
    override=False,
    asset_download_ids: List[str] = [],
    **kwargs,
):
    """A decorator to register ManiSkill environments.

    Args:
        uid (str): unique id of the environment.
        max_episode_steps (int): maximum number of steps in an episode.
        asset_download_ids (List[str]): asset download ids the environment depends on. When environments are created
            this list is checked to see if the user has all assets downloaded and if not, prompt the user if they wish to download them.
        override (bool): whether to override the environment if it is already registered.

    Notes:
        - `max_episode_steps` is processed differently from other keyword arguments in gym.
          `gym.make` wraps the env with `gym.wrappers.TimeLimit` to limit the maximum number of steps.
        - `gym.EnvSpec` uses kwargs instead of **kwargs!
    """
    try:
        json.dumps(kwargs)
    except TypeError:
        raise RuntimeError(
            f"You cannot register_env with non json dumpable kwargs, e.g. classes or types. If you really need to do this, it is recommended to create a mapping of string to the unjsonable data and to pass the string in the kwarg and during env creation find the data you need"
        )

    def _register_env(cls):
        if uid in REGISTERED_ENVS:
            if override:
                from gymnasium.envs.registration import registry

                logger.warn(f"Override registered env {uid}")
                REGISTERED_ENVS.pop(uid)
                registry.pop(uid)
            else:
                logger.warn(f"Env {uid} is already registered. Skip registration.")
                return cls

        # Register for ManiSkill
        register(
            uid,
            cls,
            max_episode_steps=max_episode_steps,
            asset_download_ids=asset_download_ids,
            default_kwargs=deepcopy(kwargs),
        )

        # Register for gym
        gym.register(
            uid,
            entry_point=partial(make, env_id=uid),
            vector_entry_point=partial(make_vec, env_id=uid),
            max_episode_steps=max_episode_steps,
            disable_env_checker=True,  # Temporary solution as we allow empty observation spaces
            kwargs=deepcopy(kwargs),
            additional_wrappers=[
                WrapperSpec(
                    "MSTimeLimit",
                    entry_point="mani_skill.utils.registration:TimeLimitWrapper",
                    kwargs=dict(max_episode_steps=max_episode_steps),
                )
            ],
        )

        return cls

    return _register_env
