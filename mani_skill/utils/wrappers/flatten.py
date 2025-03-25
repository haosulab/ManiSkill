import copy
from typing import Dict

import gymnasium as gym
import gymnasium.spaces.utils
import numpy as np
import torch
from gymnasium.vector.utils import batch_space

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common


class FlattenRGBDObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "rgbd" and "state"

    Args:
        rgb (bool): Whether to include rgb images in the observation
        depth (bool): Whether to include depth images in the observation
        state (bool): Whether to include state data in the observation
        sep_depth (bool): Whether to separate depth and rgb images in the observation. Default is True.

    Note that the returned observations will have a "rgb" or "depth" key depending on the rgb/depth bool flags, and will
    always have a "state" key. If sep_depth is False, rgb and depth will be merged into a single "rgbd" key.
    If the environment's obs_mode includes "asymmetric", state will be split into "actor-state" and "critic-state" keys.
    """

    def __init__(self, env, rgb=True, depth=True, state=True, sep_depth=True) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.sep_depth = sep_depth
        self.include_state = state

        # check if rgb/depth data exists in first camera's sensor data
        first_cam = next(iter(self.base_env._init_raw_obs["sensor_data"].values()))
        if "depth" not in first_cam:
            self.include_depth = False
        if "rgb" not in first_cam:
            self.include_rgb = False
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def _separate_actor_critic_state(
        self, observation: Dict
    ) -> tuple[Dict, Dict, Dict]:
        """Recursively separate observation into base, actor, and critic states.

        Args:
            observation: Dictionary of observations that may contain nested actor/critic states

        Returns:
            tuple[Dict, Dict, Dict]: (base_state, actor_state, critic_state)
        """
        base_state = {}
        actor_state = {}
        critic_state = {}

        for k, v in observation.items():
            if k == "actor":
                actor_state.update(v)
            elif k == "critic":
                critic_state.update(v)
            elif isinstance(v, dict):
                # Recursively process nested dictionaries
                (
                    nested_base,
                    nested_actor,
                    nested_critic,
                ) = self._separate_actor_critic_state(v)
                base_state[k] = nested_base
                actor_state.update(nested_actor)
                critic_state.update(nested_critic)
            else:
                # Leaf node - add to base state
                base_state[k] = v

        return base_state, actor_state, critic_state

    def observation(self, observation: Dict):
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        rgb_images = []
        depth_images = []
        for cam_data in sensor_data.values():
            if self.include_rgb:
                rgb_images.append(cam_data["rgb"])
            if self.include_depth:
                depth_images.append(cam_data["depth"])

        if len(rgb_images) > 0:
            rgb_images = torch.concat(rgb_images, axis=-1)
        if len(depth_images) > 0:
            depth_images = torch.concat(depth_images, axis=-1)

        # Handle state data
        if self.include_state:
            if self.base_env.obs_mode_struct.asymmetric:
                # Recursively separate actor and critic states
                (
                    base_state,
                    actor_state,
                    critic_state,
                ) = self._separate_actor_critic_state(observation)

                # Flatten base state
                base_state = common.flatten_state_dict(
                    base_state, use_torch=True, device=self.base_env.device
                )
                # Add actor-specific state if it exist
                actor_extra = common.flatten_state_dict(
                    actor_state, use_torch=True, device=self.base_env.device
                )
                actor_state = torch.cat([base_state.clone(), actor_extra], dim=-1)

                # Add critic-specific state if it exists
                critic_extra = common.flatten_state_dict(
                    critic_state, use_torch=True, device=self.base_env.device
                )
                critic_state = torch.cat([base_state.clone(), critic_extra], dim=-1)
            else:
                # Original behavior - flatten all state data
                observation = common.flatten_state_dict(
                    observation, use_torch=True, device=self.base_env.device
                )

        ret = dict()
        if self.include_state:
            if self.base_env.obs_mode_struct.asymmetric:
                ret["actor-state"] = actor_state
                ret["critic-state"] = critic_state
            else:
                ret["state"] = observation
        if self.include_rgb and not self.include_depth:
            ret["rgb"] = rgb_images
        elif self.include_rgb and self.include_depth:
            if self.sep_depth:
                ret["rgb"] = rgb_images
                ret["depth"] = depth_images
            else:
                ret["rgbd"] = torch.concat([rgb_images, depth_images], axis=-1)
        elif self.include_depth and not self.include_rgb:
            ret["depth"] = depth_images
        return ret


class FlattenObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the observations into a single vector
    """

    def __init__(self, env) -> None:
        super().__init__(env)
        self.base_env.update_obs_space(
            common.flatten_state_dict(self.base_env._init_raw_obs)
        )

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def observation(self, observation):
        return common.flatten_state_dict(observation, use_torch=True)


class FlattenActionSpaceWrapper(gym.ActionWrapper):
    """
    Flattens the action space. The original action space must be spaces.Dict
    """

    def __init__(self, env) -> None:
        super().__init__(env)
        self._orig_single_action_space = copy.deepcopy(
            self.base_env.single_action_space
        )
        self.single_action_space = gymnasium.spaces.utils.flatten_space(
            self.base_env.single_action_space
        )
        if self.base_env.num_envs > 1:
            self.action_space = batch_space(
                self.single_action_space, n=self.base_env.num_envs
            )
        else:
            self.action_space = self.single_action_space

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def action(self, action):
        if (
            self.base_env.num_envs == 1
            and action.shape == self.single_action_space.shape
        ):
            action = common.batch(action)

        # TODO (stao): This code only supports flat dictionary at the moment
        unflattened_action = dict()
        start, end = 0, 0
        for k, space in self._orig_single_action_space.items():
            end += space.shape[0]
            unflattened_action[k] = action[:, start:end]
            start += space.shape[0]
        return unflattened_action
