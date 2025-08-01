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
    """

    def __init__(self, env, rgb=True, depth=True, state=True, sep_depth=True, include_camera_params=False, include_segmentation=False) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_segmentation = include_segmentation
        self.sep_depth = sep_depth
        self.include_state = state
        self.include_camera_params = include_camera_params

        # check if rgb/depth data exists in first camera's sensor data
        first_cam = next(iter(self.base_env._init_raw_obs["sensor_data"].values()))
        if "depth" not in first_cam:
            self.include_depth = False
        if "rgb" not in first_cam:
            self.include_rgb = False
        if "segmentation" not in first_cam:
            self.include_segmentation = False
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        sensor_data = observation.pop("sensor_data")
        # Pop sensor parameters from the raw observation so they are not flattened
        sensor_param = observation.pop("sensor_param", None)

        rgb_images = []
        depth_images = []
        segmentation_images = []
        for cam_data in sensor_data.values():
            if self.include_rgb:
                rgb_images.append(cam_data["rgb"])
            if self.include_depth:
                depth_images.append(cam_data["depth"])
            if self.include_segmentation:
                segmentation_images.append(cam_data["segmentation"])
        if len(rgb_images) > 0:
            rgb_images = torch.concat(rgb_images, axis=-1)
        if len(depth_images) > 0:
            depth_images = torch.concat(depth_images, axis=-1)
        if len(segmentation_images) > 0:
            segmentation_images = torch.concat(segmentation_images, axis=-1)
        # flatten the rest of the data which should just be state data
        observation = common.flatten_state_dict(
            observation, use_torch=True, device=self.base_env.device
        )
        ret = dict()
        if self.include_state:
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
        if self.include_camera_params and sensor_param is not None:
            ret["sensor_param"] = sensor_param
        if self.include_segmentation and len(segmentation_images) > 0:
            ret["segmentation"] = segmentation_images
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
