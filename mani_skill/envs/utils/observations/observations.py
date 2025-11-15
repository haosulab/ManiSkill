"""
Functions that map a observation to a particular format, e.g. mapping the raw images to rgbd or pointcloud formats
"""


import numpy as np
import sapien.physx as physx
import torch

from mani_skill.render import SAPIEN_RENDER_SYSTEM
from mani_skill.sensors.base_sensor import BaseSensor, BaseSensorConfig
from mani_skill.sensors.camera import Camera
from mani_skill.utils import common


def sensor_data_to_pointcloud(observation: dict, sensors: dict[str, BaseSensor]):
    """convert all camera data in sensor to pointcloud data"""
    sensor_data = observation["sensor_data"]
    camera_params = observation["sensor_param"]
    pointcloud_obs = dict()

    for (cam_uid, images), (sensor_uid, sensor) in zip(
        sensor_data.items(), sensors.items()
    ):
        assert cam_uid == sensor_uid
        if isinstance(sensor, Camera):
            cam_pcd = {}
            # TODO: double check if the .clone()s are necessary
            # Each pixel is (x, y, z, actor_id) in OpenGL camera space
            # actor_id = 0 for the background
            images: dict[str, torch.Tensor]
            position = images["position"].clone()
            segmentation = images["segmentation"].clone()
            position = position.float()
            position[..., :3] = (
                position[..., :3] / 1000.0
            )  # convert the raw depth from millimeters to meters

            # Convert to world space
            cam2world = camera_params[cam_uid]["cam2world_gl"].to(position.device)
            xyzw = torch.cat([position, segmentation != 0], dim=-1).reshape(
                position.shape[0], -1, 4
            ) @ cam2world.transpose(1, 2)
            cam_pcd["xyzw"] = xyzw

            # Extra keys
            if "rgb" in images:
                rgb = images["rgb"][..., :3].clone()
                cam_pcd["rgb"] = rgb.reshape(rgb.shape[0], -1, 3)
            if "segmentation" in images:
                cam_pcd["segmentation"] = segmentation.reshape(
                    segmentation.shape[0], -1, 1
                )

            pointcloud_obs[cam_uid] = cam_pcd
    for k in pointcloud_obs.keys():
        del observation["sensor_data"][k]
    pointcloud_obs = common.merge_dicts(pointcloud_obs.values())
    for key, value in pointcloud_obs.items():
        pointcloud_obs[key] = torch.concat(value, axis=1)
    observation["pointcloud"] = pointcloud_obs

    # if not physx.is_gpu_enabled():
    #     observation["pointcloud"]["segmentation"] = (
    #         observation["pointcloud"]["segmentation"].numpy().astype(np.uint16)
    #     )
    return observation
