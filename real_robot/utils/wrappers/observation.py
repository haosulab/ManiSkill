from collections import OrderedDict
from copy import deepcopy
from typing import Sequence

import gym
import numpy as np
from gym import spaces

from real_robot.utils.common import (
    flatten_dict_keys,
    flatten_dict_space_keys,
    merge_dicts,
)


class RGBDObservationWrapper(gym.ObservationWrapper):
    """Map RealSense camera capture to rgb and depth."""

    def __init__(self, env, obs_mode="rgbd"):
        """
        :param obs_mode: if obs_mode == 'rgb', use only Color
        """
        super().__init__(env)
        self.obs_mode = obs_mode
        self.observation_space = deepcopy(env.observation_space)
        # Remove Position from camera obs space
        if self.obs_mode == "rgb":
            [cam_space.spaces.pop("depth", None)
             for cam_space in self.observation_space["image"].spaces.values()]

    def observation(self, observation: dict):
        image_obs = observation["image"]
        if self.obs_mode == "rgb":
            [cam_obs.pop("depth", None) for cam_obs in image_obs.values()]

        return observation


def merge_dict_spaces(dict_spaces: Sequence[spaces.Dict]):
    reverse_spaces = merge_dicts([x.spaces for x in dict_spaces])
    for key in reverse_spaces:
        low, high = [], []
        for x in reverse_spaces[key]:
            assert isinstance(x, spaces.Box), type(x)
            low.append(x.low)
            high.append(x.high)
        low = np.concatenate(low)
        high = np.concatenate(high)
        new_space = spaces.Box(low=low, high=high, dtype=low.dtype)
        reverse_spaces[key] = new_space
    return spaces.Dict(OrderedDict(reverse_spaces))


class PointCloudObservationWrapper(gym.ObservationWrapper):
    """Convert Position textures to world-space point cloud."""

    def __init__(self, env):
        raise NotImplementedError("Check impl for XArmBaseEnv")

        super().__init__(env)
        self.observation_space = deepcopy(env.observation_space)
        self.update_observation_space(self.observation_space)
        self._buffer = {}

    @staticmethod
    def update_observation_space(space: spaces.Dict):
        # Replace image observation spaces with point cloud ones
        image_space: spaces.Dict = space.spaces.pop("image")
        space.spaces.pop("camera_param")
        pcd_space = OrderedDict()

        for cam_uid in image_space:
            cam_image_space = image_space[cam_uid]
            cam_pcd_space = OrderedDict()

            h, w = cam_image_space["Position"].shape[:2]
            cam_pcd_space["xyzw"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(h * w, 4), dtype=np.float32
            )

            # Extra keys
            if "Color" in cam_image_space.spaces:
                cam_pcd_space["rgb"] = spaces.Box(
                    low=0, high=255, shape=(h * w, 3), dtype=np.uint8
                )
            if "Segmentation" in cam_image_space.spaces:
                cam_pcd_space["Segmentation"] = spaces.Box(
                    low=0, high=(2**32 - 1), shape=(h * w, 4), dtype=np.uint32
                )

            pcd_space[cam_uid] = spaces.Dict(cam_pcd_space)

        pcd_space = merge_dict_spaces(pcd_space.values())
        space.spaces["pointcloud"] = pcd_space

    def observation(self, observation: dict):
        image_obs = observation.pop("image")
        camera_params = observation.pop("camera_param")
        pointcloud_obs = OrderedDict()

        for cam_uid, images in image_obs.items():
            cam_pcd = {}

            # Each pixel is (x, y, z, z_buffer_depth) in OpenGL camera space
            position = images["Position"]
            # position[..., 3] = position[..., 3] < 1
            position[..., 3] = position[..., 2] < 0

            # Convert to world space
            cam2world = camera_params[cam_uid]["cam2world_gl"]
            xyzw = position.reshape(-1, 4) @ cam2world.T
            cam_pcd["xyzw"] = xyzw

            # Extra keys
            if "Color" in images:
                rgb = images["Color"][..., :3]
                rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
                cam_pcd["rgb"] = rgb.reshape(-1, 3)
            if "Segmentation" in images:
                cam_pcd["Segmentation"] = images["Segmentation"].reshape(-1, 4)

            pointcloud_obs[cam_uid] = cam_pcd

        pointcloud_obs = merge_dicts(pointcloud_obs.values())
        for key, value in pointcloud_obs.items():
            buffer = self._buffer.get(key, None)
            pointcloud_obs[key] = np.concatenate(value, out=buffer)
            self._buffer[key] = pointcloud_obs[key]

        observation["pointcloud"] = pointcloud_obs
        return observation


class RobotSegmentationObservationWrapper(gym.ObservationWrapper):
    """Add a binary mask for robot links."""

    def __init__(self, env, replace=True):
        raise NotImplementedError("Check impl for XArmBaseEnv")

        super().__init__(env)
        self.observation_space = deepcopy(env.observation_space)
        self.init_observation_space(self.observation_space, replace=replace)
        self.replace = replace
        # Cache robot link ids
        self.robot_link_ids = self.env.robot_link_ids

    @staticmethod
    def init_observation_space(space: spaces.Dict, replace: bool):
        # Update image observation spaces
        if "image" in space.spaces:
            image_space = space["image"]
            for cam_uid in image_space:
                cam_space = image_space[cam_uid]
                if "Segmentation" not in cam_space.spaces:
                    continue
                height, width = cam_space["Segmentation"].shape[:2]
                new_space = spaces.Box(
                    low=0, high=1, shape=(height, width, 1), dtype="bool"
                )
                if replace:
                    cam_space.spaces.pop("Segmentation")
                cam_space.spaces["robot_seg"] = new_space

        # Update pointcloud observation spaces
        if "pointcloud" in space.spaces:
            pcd_space = space["pointcloud"]
            if "Segmentation" in pcd_space.spaces:
                n = pcd_space["Segmentation"].shape[0]
                new_space = spaces.Box(low=0, high=1, shape=(n, 1), dtype="bool")
                if replace:
                    pcd_space.spaces.pop("Segmentation")
                pcd_space.spaces["robot_seg"] = new_space

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.robot_link_ids = self.env.robot_link_ids
        return self.observation(observation)

    def observation_image(self, observation: dict):
        image_obs = observation["image"]
        for cam_images in image_obs.values():
            if "Segmentation" not in cam_images:
                continue
            seg = cam_images["Segmentation"]
            robot_seg = np.isin(seg[..., 1:2], self.robot_link_ids)
            if self.replace:
                cam_images.pop("Segmentation")
            cam_images["robot_seg"] = robot_seg
        return observation

    def observation_pointcloud(self, observation: dict):
        pointcloud_obs = observation["pointcloud"]
        if "Segmentation" not in pointcloud_obs:
            return observation
        seg = pointcloud_obs["Segmentation"]
        robot_seg = np.isin(seg[..., 1:2], self.robot_link_ids)
        if self.replace:
            pointcloud_obs.pop("Segmentation")
        pointcloud_obs["robot_seg"] = robot_seg
        return observation

    def observation(self, observation: dict):
        if "image" in observation:
            observation = self.observation_image(observation)
        if "pointcloud" in observation:
            observation = self.observation_pointcloud(observation)
        return observation


class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env) -> None:
        raise NotImplementedError("Check impl for XArmBaseEnv")

        super().__init__(env)
        self.observation_space = flatten_dict_space_keys(self.observation_space)

    def observation(self, observation):
        return flatten_dict_keys(observation)
