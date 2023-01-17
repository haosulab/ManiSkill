from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Sequence

import gym
import numpy as np
import torch
from gym import spaces

from mani_skill2.utils.common import merge_dicts

from ..vec_env import VecEnvObservationWrapper


class RGBDObservationWrapper(VecEnvObservationWrapper):
    """Map raw textures (Color and Position) to rgb and depth."""

    def __init__(self, venv):
        super().__init__(venv)
        self.observation_space = deepcopy(venv.observation_space)

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
                    rgb = torch.clamp(ori_images[key][..., :3] * 255, 0, 255)
                    rgb = rgb.to(dtype=torch.uint8)
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


class PointCloudObservationWrapper(VecEnvObservationWrapper):
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
        pointcloud_obs = defaultdict(list)
        for name, images in image_obs.items():
            # Each pixel is (x, y, z, z_buffer_depth) in OpenGL camera space
            position = images["Position"]
            # TODO(jigu): inplace?
            position[..., 3] = position[..., 3] < 1
            # Convert to world space
            cam2world = camera_params[name]["cam2world_gl"]  # OpenGL convention
            cam2world = torch.from_numpy(cam2world).to(
                device=position.device, non_blocking=True
            )
            bs = position.size(0)
            xyzw = torch.bmm(position.reshape(bs, -1, 4), cam2world.transpose(1, 2))
            pointcloud_obs["xyzw"].append(xyzw)

            if "Color" in images:
                rgb = images["Color"][..., :3]
                rgb = torch.clamp(rgb * 255, 0, 255).to(torch.uint8, non_blocking=True)
                pointcloud_obs["rgb"].append(rgb.reshape(bs, -1, 3))

            if "robot_seg" in images:
                robot_seg = images["robot_seg"]
                pointcloud_obs["robot_seg"].append(robot_seg.reshape(bs, -1, 1))

        for key, value in pointcloud_obs.items():
            pointcloud_obs[key] = torch.cat(value, dim=1)
        observation["pointcloud"] = pointcloud_obs
        return observation


class RobotSegmentationObservationWrapper(VecEnvObservationWrapper):
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
        self.update_robot_link_ids()

    def update_robot_link_ids(self):
        self.robot_link_ids = torch.tensor(
            self.venv.get_attr("robot_link_ids"), dtype=torch.int32
        ).to(device=self._obs_tensors[0].device, non_blocking=True)

    def reset_wait(self, **kwargs):
        observation = self.venv.reset_wait(**kwargs)
        self.update_robot_link_ids()
        return self.observation(observation)

    def observation(self, observation: dict):
        image_obs = observation["image"]
        for name, ori_images in image_obs.items():
            if "Segmentation" not in ori_images:
                continue
            seg = ori_images["Segmentation"]

            # # For-loop version
            # robot_seg_batch = []
            # for i in range(self.num_envs):
            #     robot_seg = torch.isin(seg[i][..., 1:2], self.robot_link_ids[i])
            #     robot_seg_batch.append(robot_seg)
            # robot_seg_batch = torch.stack(robot_seg_batch, dim=0)

            actor_seg = seg[..., 1:2]  # [B, H, W, 1]
            bs = actor_seg.size(0)
            max_ids, _ = torch.max(actor_seg.reshape(bs, -1), dim=-1)  # [B]
            max_ids = torch.nn.functional.pad(max_ids[:-1], (1, 0), value=0)
            actor_seg = actor_seg + max_ids.view(bs, 1, 1, 1)
            robot_link_ids = self.robot_link_ids + max_ids.view(bs, 1)
            robot_seg_batch = torch.isin(actor_seg, robot_link_ids)

            if self.replace:
                ori_images.pop("Segmentation")
            ori_images["robot_seg"] = robot_seg_batch
        return observation
