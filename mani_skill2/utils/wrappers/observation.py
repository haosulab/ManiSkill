from collections import OrderedDict
from copy import deepcopy
from typing import Sequence

import gym
import numpy as np
from gym import spaces

from mani_skill2.utils.common import merge_dicts


class RGBDObservationWrapper(gym.ObservationWrapper):
    """Map raw textures (Color and Position) to rgb and depth."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = deepcopy(env.observation_space)

        # Update image observation spaces
        image_obs_spaces: spaces.Dict = self.observation_space.spaces["image"]
        for name in image_obs_spaces:
            ori_obs_spaces = image_obs_spaces[name]
            new_obs_spaces = OrderedDict()
            for key in ori_obs_spaces:
                if key == "Color":
                    height, width = ori_obs_spaces[key].shape[:2]
                    new_obs_spaces["rgb"] = spaces.Box(
                        low=0, high=255, shape=(height, width, 3), dtype=np.uint8
                    )
                elif key == "Position":
                    height, width = ori_obs_spaces[key].shape[:2]
                    new_obs_spaces["depth"] = spaces.Box(
                        low=0, high=np.inf, shape=(height, width, 1), dtype=np.float32
                    )
                else:
                    new_obs_spaces[key] = ori_obs_spaces[key]
            image_obs_spaces.spaces[name] = spaces.Dict(new_obs_spaces)

    def observation(self, observation: dict):
        image_obs = observation["image"]
        for name, ori_images in image_obs.items():
            new_images = OrderedDict()
            for key in ori_images:
                if key == "Color":
                    rgb = np.clip(ori_images[key][..., :3] * 255, 0, 255).astype(
                        np.uint8
                    )
                    new_images["rgb"] = rgb
                elif key == "Position":
                    depth = -ori_images[key][..., [2]]  # [H, W, 1]
                    new_images["depth"] = depth
                else:
                    new_images[key] = ori_images[key]
            image_obs[name] = new_images
        return observation


def merge_dict_spaces(dict_spaces: Sequence[spaces.Dict]):
    reverse_spaces = merge_dicts([x.spaces for x in dict_spaces])
    for key in reverse_spaces:
        low, high = [], []
        for x in reverse_spaces[key]:
            assert isinstance(x, spaces.Box), type(x)
            low.append(x.low)
            high.append(x.high)
        new_space = spaces.Box(low=np.concatenate(low), high=np.concatenate(high))
        reverse_spaces[key] = new_space
    return spaces.Dict(OrderedDict(reverse_spaces))


class PointCloudObservationWrapper(gym.ObservationWrapper):
    """Convert Position textures to world-space point cloud."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = deepcopy(env.observation_space)

        # Replace image observation spaces with point cloud ones
        image_obs_spaces: spaces.Dict = self.observation_space.spaces.pop("image")
        self.observation_space.spaces.pop("camera_param")
        pcd_obs_spaces = OrderedDict()
        for name in image_obs_spaces:
            ori_obs_spaces = image_obs_spaces[name]
            new_obs_spaces = OrderedDict()

            h, w = ori_obs_spaces["Position"].shape[:2]
            new_obs_spaces["xyzw"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(h * w, 4), dtype=np.float32
            )

            if "Color" in ori_obs_spaces:
                new_obs_spaces["rgb"] = spaces.Box(
                    low=0, high=255, shape=(h * w, 3), dtype=np.uint8
                )

            if "robot_seg" in ori_obs_spaces:
                new_obs_spaces["robot_seg"] = spaces.Box(
                    low=0, high=1, shape=(h * w, 1), dtype=np.bool_
                )

            pcd_obs_spaces[name] = spaces.Dict(new_obs_spaces)

        pcd_obs_spaces = merge_dict_spaces(pcd_obs_spaces.values())
        self.observation_space.spaces["pointcloud"] = pcd_obs_spaces

    def observation(self, observation: dict):
        image_obs = observation.pop("image")
        camera_params = observation.pop("camera_param")
        pointcloud_obs = OrderedDict()
        for name, images in image_obs.items():
            pcds = {}

            # Each pixel is (x, y, z, z_buffer_depth) in OpenGL camera space
            position = images["Position"]
            position[..., 3] = position[..., 3] < 1
            # Convert to world space
            cam2world = camera_params[name]["cam2world_gl"]  # OpenGL convention
            xyzw = position.reshape(-1, 4) @ cam2world.T
            pcds["xyzw"] = xyzw

            if "Color" in images:
                rgb = images["Color"][..., :3]
                rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
                pcds["rgb"] = rgb.reshape(-1, 3)

            if "robot_seg" in images:
                pcds["robot_seg"] = images["robot_seg"].reshape(-1, 1)

            pointcloud_obs[name] = pcds

        pointcloud_obs = merge_dicts(pointcloud_obs.values(), asarray=True)
        observation["pointcloud"] = pointcloud_obs
        return observation


class RobotSegmentationObservationWrapper(gym.ObservationWrapper):
    """Add a binary mask for robot links."""

    def __init__(self, env, replace=True):
        super().__init__(env)
        self.observation_space = deepcopy(env.observation_space)
        self.replace = replace

        # Update image observation spaces
        image_obs_spaces: spaces.Dict = self.observation_space.spaces["image"]
        for name in image_obs_spaces:
            ori_obs_spaces = image_obs_spaces[name]
            if "Segmentation" not in ori_obs_spaces.spaces:
                continue
            height, width = ori_obs_spaces["Segmentation"].shape[:2]
            new_obs_space = spaces.Box(
                low=0, high=1, shape=(height, width, 1), dtype=np.bool_
            )
            if self.replace:
                ori_obs_spaces.spaces.pop("Segmentation")
            ori_obs_spaces.spaces["robot_seg"] = new_obs_space

        # Cache robot link ids
        self.robot_link_ids = self.env.robot_link_ids

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.robot_link_ids = self.env.robot_link_ids
        return self.observation(observation)

    def observation(self, observation: dict):
        image_obs = observation["image"]
        for name, ori_images in image_obs.items():
            if "Segmentation" not in ori_images:
                continue
            seg = ori_images["Segmentation"]
            robot_seg = np.isin(seg[..., 1:2], self.robot_link_ids)
            robot_seg = robot_seg
            if self.replace:
                ori_images.pop("Segmentation")
            ori_images["robot_seg"] = robot_seg
        return observation
