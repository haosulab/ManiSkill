"""
Functions that map a observation to a particular format, e.g. mapping the raw images to rgbd or pointcloud formats
"""

from typing import Dict

import numpy as np
import sapien.physx as physx
import torch

from mani_skill.sensors.base_sensor import BaseSensor, BaseSensorConfig
from mani_skill.sensors.camera import Camera
from mani_skill.utils import common
from mani_skill.envs.utils.observations.voxelizer import VoxelGrid

def sensor_data_to_rgbd(
    observation: Dict,
    sensors: Dict[str, BaseSensor],
    rgb=True,
    depth=True,
    segmentation=True,
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
            new_images = dict()
            ori_images: Dict[str, torch.Tensor]
            for key in ori_images:
                if key == "Color":
                    if rgb:
                        rgb_data = ori_images[key][..., :3].clone()  # [H, W, 4]
                        new_images["rgb"] = rgb_data  # [H, W, 4]
                elif key == "PositionSegmentation":
                    if depth:
                        depth_data = -ori_images[key][..., [2]]  # [H, W, 1]
                        # NOTE (stao): This is a bit of a hack since normally we have generic to_numpy call to convert
                        # internal torch tensors to numpy if we do not use GPU simulation
                        # but torch does not have a uint16 type so we convert that here earlier
                        # if not physx.is_gpu_enabled():
                        #     depth_data = depth_data.numpy().astype(np.uint16)
                        new_images["depth"] = depth_data
                    if segmentation:
                        segmentation_data = ori_images[key][..., [3]]
                        # if not physx.is_gpu_enabled():
                        #     segmentation_data = segmentation_data.numpy().astype(
                        #         np.uint16
                        #     )
                        new_images["segmentation"] = segmentation_data  # [H, W, 1]
                else:
                    new_images[key] = ori_images[key]
            sensor_data[cam_uid] = new_images
    return observation


def sensor_data_to_pointcloud(observation: Dict, sensors: Dict[str, BaseSensor]):
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

            # Each pixel is (x, y, z, actor_id) in OpenGL camera space
            # actor_id = 0 for the background
            images: Dict[str, torch.Tensor]
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

def sensor_data_to_voxel(
    observation: Dict, 
    sensors: Dict[str, BaseSensor],
    obs_mode_config: Dict
    ):
    """convert all camera data in sensor to voxel grid"""
    sensor_data = observation["sensor_data"]
    camera_params = observation["sensor_param"]
    coord_bounds = obs_mode_config["coord_bounds"] # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
    voxel_size = obs_mode_config["voxel_size"] # size of the voxel grid (assuming cubic)
    device = obs_mode_config["device"] # device on which doing voxelization
    seg = obs_mode_config["segmentation"] # device on which doing voxelization
    pcd_rgb_observations = dict()

    # Collect all cameras' observations
    for (cam_uid, images), (sensor_uid, sensor) in zip(
        sensor_data.items(), sensors.items()
    ):
        assert cam_uid == sensor_uid
        if isinstance(sensor, Camera):
            cam_data = {}
            
            # Extract point cloud and segmentation data
            images: Dict[str, torch.Tensor]
            position = images["PositionSegmentation"]
            if seg:
                segmentation = position[..., 3].clone()
            position = position.float()
            position[..., 3] = 1 # convert to homogeneious coordinates
            position[..., :3] = (
                position[..., :3] / 1000.0
            )  # convert the raw depth from millimeters to meters

            # Convert to world space position and update camera data
            cam2world = camera_params[cam_uid]["cam2world_gl"]
            xyzw = position.reshape(position.shape[0], -1, 4) @ cam2world.transpose(
                1, 2
            )
            xyz = xyzw[..., :3] / xyzw[..., 3].unsqueeze(-1) # dehomogeneize
            cam_data["xyz"] = xyz
            if seg:
                cam_data["seg"] = segmentation.reshape(segmentation.shape[0], -1, 1)

            # Extract rgb data
            if "Color" in images:
                rgb = images["Color"][..., :3].clone()
                rgb = rgb / 255 # convert to range [0, 1]
                cam_data["rgb"] = rgb.reshape(rgb.shape[0], -1, 3)

            pcd_rgb_observations[cam_uid] = cam_data

    # just free sensor_data to save memory
    for k in pcd_rgb_observations.keys():
        del observation["sensor_data"][k]

    # merge features from different cameras together
    pcd_rgb_observations = common.merge_dicts(pcd_rgb_observations.values())
    for key, value in pcd_rgb_observations.items():
        pcd_rgb_observations[key] = torch.concat(value, axis=1)
    
    # prepare features for voxel convertions
    xyz_dev = pcd_rgb_observations["xyz"].to(device)    
    rgb_dev = pcd_rgb_observations["rgb"].to(device)    
    if seg:
        seg_dev = pcd_rgb_observations["seg"].to(device)    
    coord_bounds = torch.tensor(coord_bounds, device=device).unsqueeze(0)
    batch_size = xyz_dev.shape[0]
    max_num_coords = rgb_dev.shape[1]
    vox_grid = VoxelGrid(
        coord_bounds=coord_bounds,
        voxel_size=voxel_size,
        device=device,
        batch_size=batch_size,
        feature_size=3,
        max_num_coords=max_num_coords,
    )

    # convert to the batched voxel grids
    # voxel 11D features contain: 3 (pcd xyz coordinates) + 3 (rgb) + 3 (voxel xyz indices) + 1 (seg id if applicable) + 1 (occupancy)
    if seg: # add voxel segmentations
        voxel_grid = vox_grid.coords_to_bounding_voxel_grid(xyz_dev,
                                                    coord_features=rgb_dev,
                                                    coord_bounds=coord_bounds,
                                                    clamp_vox_id=True, 
                                                    pcd_seg=seg_dev) 
    else: # no voxel segmentation
        voxel_grid = vox_grid.coords_to_bounding_voxel_grid(xyz_dev,
                                                    coord_features=rgb_dev,
                                                    coord_bounds=coord_bounds,
                                                    clamp_vox_id=False) 

    # update voxel grids to the observation dict
    observation["voxel_grid"] = voxel_grid 
    return observation