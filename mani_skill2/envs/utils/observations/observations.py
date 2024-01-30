"""
Functions that map a observation to a particular format, e.g. mapping the raw images to rgbd or pointcloud formats
"""

from collections import OrderedDict
from typing import Dict

import numpy as np
import torch

from mani_skill2.utils.common import merge_dicts


def image_to_rgbd(observation: Dict):
    image_obs = observation["image"]
    for cam_uid, ori_images in image_obs.items():
        new_images = OrderedDict()
        for key in ori_images:
            if key == "Color":
                rgb = ori_images[key][..., :3].clone()  # [H, W, 4]
                new_images["rgb"] = rgb  # [H, W, 4]
            elif key == "PositionSegmentation":
                depth = -ori_images[key][..., [2]]  # [H, W, 1]
                # NOTE (stao): This is a bit of a hack since normally we have generic to_numpy call to convert internal torch tensors to numpy if we do not use GPU simulation
                # but torch does not have a uint16 type so we convert that here earlier
                if depth.shape[0] == 1:
                    depth = depth.numpy().astype(np.uint16)
                new_images["depth"] = depth
            else:
                new_images[key] = ori_images[key]
        image_obs[cam_uid] = new_images
    return observation


def image_to_pointcloud(observation: Dict):
    image_obs = observation.pop("image")
    camera_params = observation.pop("camera_param")
    pointcloud_obs = OrderedDict()

    for cam_uid, images in image_obs.items():
        cam_pcd = {}

        # Each pixel is (x, y, z, actor_id) in OpenGL camera space
        # actor_id = 0 for the background
        position = images["PositionSegmentation"]
        segmentation = position[..., 3].clone()
        position[..., 3] = position[..., 2] == 0
        position = position / 1000.0

        # Convert to world space
        cam2world = camera_params[cam_uid]["cam2world_gl"]
        xyzw = position.reshape(position.shape[0], -1, 4) @ cam2world.transpose(1, 2)
        cam_pcd["xyzw"] = xyzw

        # Extra keys
        if "Color" in images:
            rgb = images["Color"][..., :3].clone()
            cam_pcd["rgb"] = rgb.reshape(rgb.shape[0], -1, 3)
        if "PositionSegmentation" in images:
            cam_pcd["Segmentation"] = segmentation.reshape(segmentation.shape[0], -1)

        pointcloud_obs[cam_uid] = cam_pcd

    pointcloud_obs = merge_dicts(pointcloud_obs.values())
    for key, value in pointcloud_obs.items():
        pointcloud_obs[key] = torch.concat(value)

    observation["pointcloud"] = pointcloud_obs
    return observation


# TODO (stao):

# def image_to_rgbd_robot_seg(observation: Dict):
#     ...

# def image_to_pointcloud_robot_seg(observation: Dict):
#     ...


# TODO (stao): add segmentation ids
