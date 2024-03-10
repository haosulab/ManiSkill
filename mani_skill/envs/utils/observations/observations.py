"""
Functions that map a observation to a particular format, e.g. mapping the raw images to rgbd or pointcloud formats
"""

from collections import OrderedDict
from typing import Dict

import numpy as np
import sapien.physx as physx
import torch

from mani_skill.sensors.base_sensor import BaseSensor, BaseSensorConfig
from mani_skill.sensors.camera import Camera
from mani_skill.utils.common import merge_dicts


def sensor_data_to_rgbd(
    observation: Dict,
    sensors: Dict[str, BaseSensor],
    rgb=True,
    depth=True,
    segmentation: str = "",
):
    """
    Converts all camera data to a easily usable rgb+depth format

    Optionally can include segmentation
    """
    sensor_data = observation["sensor_data"]
    for (cam_uid, ori_images), (sensor_uid, sensor) in zip(
        sensor_data.items(), sensors.items()
    ):
        assert cam_uid == sensor_uid
        if isinstance(sensor, Camera):
            new_images = OrderedDict()
            ori_images: dict[str, torch.Tensor]
            for key in ori_images:
                if key == "Color":
                    if rgb:
                        rgb_data = ori_images[key][..., :3].clone()  # [H, W, 4]
                        new_images["rgb"] = rgb_data  # [H, W, 4]
                elif key == "PositionSegmentation":
                    if depth:
                        depth_data = -ori_images[key][..., [2]]  # [H, W, 1]
                        # NOTE (stao): This is a bit of a hack since normally we have generic to_numpy call to convert internal torch tensors to numpy if we do not use GPU simulation
                        # but torch does not have a uint16 type so we convert that here earlier
                        if not physx.is_gpu_enabled():
                            depth_data = depth_data.numpy().astype(np.uint16)
                        new_images["depth"] = depth_data
                else:
                    new_images[key] = ori_images[key]
            sensor_data[cam_uid] = new_images
    return observation


def sensor_data_to_pointcloud(observation: Dict, sensors: Dict[str, BaseSensor]):
    """convert all camera data in sensor to pointcloud data"""
    sensor_data = observation["sensor_data"]
    camera_params = observation["sensor_param"]
    pointcloud_obs = OrderedDict()

    for (cam_uid, images), (sensor_uid, sensor) in zip(
        sensor_data.items(), sensors.items()
    ):
        assert cam_uid == sensor_uid
        if isinstance(sensor, Camera):
            cam_pcd = {}

            # Each pixel is (x, y, z, actor_id) in OpenGL camera space
            # actor_id = 0 for the background
            images: dict[str, torch.Tensor]
            position = images["PositionSegmentation"]
            segmentation = position[..., 3].clone()
            position = position.float()
            position[..., 3] = position[..., 3] != 0
            position[..., :3] = (
                position[..., :3] / 1000.0
            )  # convert the raw depth from millimeters to meters

            # Convert to world space
            cam2world = camera_params[cam_uid]["cam2world_gl"]
            xyzw = position.reshape(position.shape[0], -1, 4) @ cam2world.transpose(
                1, 2
            )
            cam_pcd["xyzw"] = xyzw

            # Extra keys
            if "Color" in images:
                rgb = images["Color"][..., :3].clone()
                cam_pcd["rgb"] = rgb.reshape(rgb.shape[0], -1, 3)
            if "PositionSegmentation" in images:
                cam_pcd["Segmentation"] = segmentation.reshape(
                    segmentation.shape[0], -1
                )

            pointcloud_obs[cam_uid] = cam_pcd
    for k in pointcloud_obs.keys():
        del observation["sensor_data"][k]
    pointcloud_obs = merge_dicts(pointcloud_obs.values())
    for key, value in pointcloud_obs.items():
        pointcloud_obs[key] = torch.concat(value, axis=1)
    observation["pointcloud"] = pointcloud_obs
    return observation


# TODO (stao):

# def image_to_rgbd_robot_seg(observation: Dict):
#     ...

# def image_to_pointcloud_robot_seg(observation: Dict):
#     ...


# TODO (stao): add segmentation ids
