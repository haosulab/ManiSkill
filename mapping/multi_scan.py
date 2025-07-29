import argparse
import numpy as np
import open3d as o3d
import plotly.graph_objs as go
import plotly.offline as pyo
import gymnasium as gym
import imageio.v2 as imageio
from transforms3d.quaternions import mat2quat
from typing import Dict, Any, List

import sapien
from pathlib import Path
from tqdm.auto import trange, tqdm
from mani_skill.utils import visualization
import mani_skill.envs.tasks.tabletop.table_scan_discrete_no_robot

# MARK: Parse command line arguments
parser = argparse.ArgumentParser(description="Scan an object and generate a point cloud.")
parser.add_argument("-o", "--obs-mode", type=str, default="rgb", help="Can be rgb or rgb+depth, rgb+normal, albedo+depth etc. Which ever image-like textures you want to visualize can be tacked on")
parser.add_argument("-n", "--num-envs", type=int, default=32, help="Number of parallel environments to run and visualize"
)
parser.add_argument("-s","--seed",type=int, default=0, help="Seed the random actions and environment. Default is no seed",)
parser.add_argument("--total-envs", type=int, default=64, help="Total number of environments to process")
parser.add_argument("--batch-size", type=int, default=16, help="How many parallel envs to run at once")
args = parser.parse_args()

# MARK: Setup directories and paths
SCRIPT_DIR = Path(__file__).resolve().parent
POSE_DIR   = SCRIPT_DIR / "poses"
POSE_DIR.mkdir(exist_ok=True)
pose_filename = SCRIPT_DIR / "poses.txt"
OUTPUT_DIR = SCRIPT_DIR / "reloaded_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# MARK: Auxiliary functions
def load_poses_from_file(path: str) -> np.ndarray:
    """Return array of shape (T,4,4) with homogeneous cam→world matrices"""
    mats = []
    with open(path, "r") as f:
        for line in f:
            vals = np.fromstring(line, sep=" ")
            assert vals.size == 12, "Expect 12 numbers (3×4) per line"
            M = np.eye(4, dtype=np.float32)
            M[:3] = vals.reshape(3, 4)
            mats.append(M)
    return np.stack(mats, axis=0) 

if args.seed is not None:
    np.random.seed(args.seed)


# MARK: Capture images and point cloud for multiple episodes
try:
    all_poses = load_poses_from_file(pose_filename)
    print(f"Loaded {len(all_poses)} poses from {pose_filename}")
except FileNotFoundError as e:
    print(f"Error {e}")
    print("Please run the pose generation script first.")

env = gym.make(
    "TableScanDiscreteNoRobot-v0",
    robot_uids="none",
    obs_mode=args.obs_mode,
    num_envs=args.batch_size,
    control_mode="pd_joint_pos",
    reward_mode="none"
)
obs, _ = env.reset(seed=0)

all_rendered_images = []
num_poses = len(all_poses)

# The correct rotation matrix to flip from CV's +Z-forward to SAPIEN's -Z-forward
oRc = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)

def tile(frames: List[np.ndarray], nrows: int) -> np.ndarray:
    h, w, _ = frames[0].shape
    ncols   = int(np.ceil(len(frames) / nrows))
    grid    = np.zeros((nrows * h, ncols * w, 3), dtype=np.uint8)
    for idx, f in enumerate(frames):
        r, c = divmod(idx, ncols)
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = f
    return grid


# If you want to view the grid live
# renderer = visualization.ImageRenderer()


GRID_ROWS = int(np.ceil(np.sqrt(args.batch_size)))


chunk_count   = args.num_envs // args.batch_size
global_env_id = 0

for chunk_idx in range(chunk_count):
    env = gym.make(
        "TableScanDiscreteNoRobot-v0",
        robot_uids="none",
        obs_mode=args.obs_mode,
        num_envs=args.batch_size,
        control_mode="pd_joint_pos",
        reward_mode="none",
    )
    env.reset(seed=0)
    unwrapped = env.unwrapped

    grid_frames: List[np.ndarray] = []

    for pose_mat in all_poses:
        R_cv, t_cv = pose_mat[:3, :3], pose_mat[:3, 3]
        R_gl       = R_cv @ oRc
        pose       = sapien.Pose(p=t_cv, q=mat2quat(R_gl))

        unwrapped.cam_mount.set_pose(pose)
        unwrapped.scene.update_render()
        cam = unwrapped.scene.human_render_cameras["moving_camera"]
        cam.camera.take_picture()
        rgb_batch = cam.get_obs()["rgb"].cpu().numpy()

        frames = list(rgb_batch)
        grid   = tile(frames, nrows=GRID_ROWS)
        # If you want to view the grid live
        # renderer(grid) 
        grid_frames.append(grid)

    out = OUTPUT_DIR / f"grid_chunk{chunk_idx}.mp4"
    imageio.mimsave(out, grid_frames, fps=10)
    print(f"Wrote {out}")

    env.close()
    global_env_id += args.batch_size

print("Done.")