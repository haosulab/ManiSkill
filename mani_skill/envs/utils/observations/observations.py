"""
Functions that map a observation to a particular format, e.g. mapping the raw images to rgbd or pointcloud formats
"""

from typing import Dict

import numpy as np
import sapien.physx as physx
import torch

from mani_skill.render import SAPIEN_RENDER_SYSTEM
from mani_skill.sensors.base_sensor import BaseSensor, BaseSensorConfig
from mani_skill.sensors.camera import Camera
from mani_skill.utils import common
from mani_skill.envs.utils.observations.voxelizer import VoxelGrid

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
            # TODO: double check if the .clone()s are necessary
            # Each pixel is (x, y, z, actor_id) in OpenGL camera space
            # actor_id = 0 for the background
            images: Dict[str, torch.Tensor]
            position = images["position"].clone()
            segmentation = images["segmentation"].clone()
            position = position.float()
            position[..., :3] = (
                position[..., :3] / 1000.0
            )  # convert the raw depth from millimeters to meters

            # Convert to world space
            cam2world = camera_params[cam_uid]["cam2world_gl"]
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
    bb_maxs = torch.tensor(coord_bounds[3:6]) # max voxel scene boundary
    bb_maxs = bb_maxs.to(torch.float)
    bb_maxs = bb_maxs.unsqueeze(0)

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
            position[..., 3] = position[..., 3] != 0 # mask out infinitely far points
            position = position.float() # position -> camera-frame xyzw coordinates

            # Record w=0 (infinitely far) points' indices
            out_indices = (position[..., 3] == 0)
            out_indices = out_indices.reshape(out_indices.shape[0], -1)            

            # Convert to world space position and update camera data
            position[..., 3] = 1 # for matrix multiplication
            position[..., :3] = (
                position[..., :3] / 1000.0
            )  # convert the raw depth from millimeters to meters
            cam2world = camera_params[cam_uid]["cam2world_gl"]
            xyzw = position.reshape(position.shape[0], -1, 4) @ cam2world.transpose(
                1, 2
            )
            
            # Set w=0 points outside the bounds, so that they can be cropped during voxelization
            xyz = xyzw[..., :3] / xyzw[..., 3].unsqueeze(-1) # dehomogeneize
            xyz[out_indices, :] = bb_maxs + 1
            cam_data["xyz"] = xyz

            # Extract seg and rgb data 
            if seg:
                cam_data["seg"] = segmentation.reshape(segmentation.shape[0], -1, 1)
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