import argparse

import gymnasium as gym
import numpy as np

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.visualization.voxel_visualizer import visualise_voxel
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PushCube-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument("--voxel-size", type=int, default=200, help="The number of voxels per side")
    parser.add_argument("--zoom-factor", type=float, default=1, help="Zoom factor of the camera when generating the output voxel visualizations")
    parser.add_argument("--segmentation", type=bool, default=True, help="Whether or not to include voxel segmentation estimations")
    parser.add_argument("--rotation-amount", type=float, default=45, help="The amount of rotation of camera for filming the voxelized scene")
    parser.add_argument("--cam-width", type=int, default=720, help="Override the width of every camera in the environment")
    parser.add_argument("--cam-height", type=int, default=480, help="Override the height of every camera in the environment")
    parser.add_argument(
        "--coord-bounds",
        nargs="*",
        type=float,
        default=[-1, -1, -1, 2, 2, 2],
        help="The bounds of the 3D points' coordinates to be voxelized"
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and environment. Default is no seed",
    )
    args = parser.parse_args()
    return args

def render_filtered_voxels(voxel_grid, zf=1.0, rotation_amount=60):
    flood_id = 17
    vis_voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach().cpu().numpy()
    floor_map = (vis_voxel_grid[:, 9, ...] == flood_id)
    floor_map = torch.tensor(floor_map)
    floor_map = floor_map.unsqueeze(1).repeat(1, 11, 1, 1, 1)
    vis_voxel_grid[floor_map] = 0

    rendered_img = visualise_voxel(vis_voxel_grid[0],
                                None,
                                None,
                                voxel_size=0.01,
                                zoom_factor=zf,
                                rotation_amount=np.deg2rad(rotation_amount))
    return rendered_img


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
    sensor_configs = dict()
    if args.cam_width:
        sensor_configs["width"] = args.cam_width
    if args.cam_height:
        sensor_configs["height"] = args.cam_height
    obs_mode_config = {"coord_bounds": args.coord_bounds, # TODO: add a list of coord bounds by parsing them
                    "voxel_size": args.voxel_size, 
                    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    "segmentation": args.segmentation}
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode="voxel",
        reward_mode="none",
        obs_mode_config=obs_mode_config,
        sensor_configs=sensor_configs,
    )

    # Interactively show the voxelized scene
    zf = args.zoom_factor # controlling camera zoom-ins
    obs, _ = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        voxel_grid = obs["voxel_grid"]
        img = render_filtered_voxels(voxel_grid, zf, args.rotation_amount)
        plt.axis('off')
        plt.imshow(img)
        plt.show()
        if terminated or truncated:
            break
    env.close()



if __name__ == "__main__":
    main(parse_args())
