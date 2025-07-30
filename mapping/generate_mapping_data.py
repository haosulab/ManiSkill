import argparse
import numpy as np
import open3d as o3d
from transforms3d.quaternions import mat2quat
from typing import List
import sapien
from pathlib import Path
import mani_skill.envs.tasks.tabletop.table_scan_discrete_no_robot
import gymnasium as gym
import cv2
# MARK: Parse command line arguments
parser = argparse.ArgumentParser(description="Scan an object and generate a point cloud.")
parser.add_argument("-o", "--obs-mode", type=str, default="rgbd", help="Can be rgb or rgb+depth, rgb+normal, albedo+depth etc. Which ever image-like textures you want to visualize can be tacked on")
parser.add_argument("-n", "--num-envs", type=int, default=100, help="Total number of environments to process (across all batches)")
parser.add_argument("-s","--seed",type=int, default=0, help="Seed the random actions and environment. Default is no seed",)
parser.add_argument("--batch-size", type=int, default=100, help="How many parallel envs to run at once")
args = parser.parse_args()
# MARK: Setup directories and paths
SCRIPT_DIR = Path(__file__).resolve().parent
POSE_DIR   = SCRIPT_DIR / "poses"
POSE_DIR.mkdir(exist_ok=True)
pose_filename = SCRIPT_DIR / "poses.txt"
OUTPUT_DIR = SCRIPT_DIR / "reloaded_output"
OUTPUT_DIR.mkdir(exist_ok=True)
DATASET_DIR = SCRIPT_DIR / "dataset"
DATASET_DIR.mkdir(exist_ok=True)
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
K_env0 = None
for chunk_idx in range(chunk_count):
    env = gym.make(
        "TableScanDiscreteNoRobot-v0",
        robot_uids="none",
        obs_mode=args.obs_mode,
        num_envs=args.batch_size,
        control_mode="pd_joint_pos",
        reward_mode="none",
    )
    # Use a different seed for each chunk to get different objects
    env.reset(seed=args.seed + chunk_idx)
    # ------------------------------------------------------------------ #
    # Prepare output directories for the current batch
    # ------------------------------------------------------------------ #
    for i in range(args.batch_size):
        env_id = global_env_id + i
        env_dir = DATASET_DIR / f"env_{env_id:03d}"
        (env_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (env_dir / "depth").mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------------ #
    # Prepare containers for multiview point cloud aggregation
    # ------------------------------------------------------------------ #
    pts_all: List[List[np.ndarray]]  = [[] for _ in range(args.batch_size)]
    rgb_all: List[List[np.ndarray]]  = [[] for _ in range(args.batch_size)]
    unwrapped = env.unwrapped
    grid_frames: List[np.ndarray] = []
    for pose_idx, pose_mat in enumerate(all_poses):
        R_cv, t_cv = pose_mat[:3, :3], pose_mat[:3, 3]
        R_gl       = R_cv @ oRc
        pose       = sapien.Pose(p=t_cv, q=mat2quat(R_gl))
        unwrapped.cam_mount.set_pose(pose)
        unwrapped.scene.update_render()
        cam = unwrapped.scene.human_render_cameras["moving_camera"]
        # Trigger rendering
        cam.camera.take_picture()
        # ------------------------------------------------------------------ #
        # Fetch sensor data & parameters for the entire batch
        # ------------------------------------------------------------------ #
        cam_obs    = cam.get_obs()
        rgb_batch   = cam_obs["rgb"].cpu().numpy()       # (B, H, W, 3)
        depth_batch = cam_obs["depth"].cpu().numpy()      # (B, H, W)
        # Note: These matrices are of shape (B, 3, 3)/(B, 3, 4)
        K_batch  = cam.camera.get_intrinsic_matrix()
        if chunk_idx == 0 and K_env0 is None:
            K_env0 = K_batch[0].cpu().numpy()
        E_batch  = cam.camera.get_extrinsic_matrix()
        # ------------------------------------------------------------------ #
        # Accumulate RGB frames for video grid (unchanged behaviour)
        # ------------------------------------------------------------------ #
        frames = list(rgb_batch)
        grid   = tile(frames, nrows=GRID_ROWS)
        # If you want to view the grid live
        # renderer(grid)
        grid_frames.append(grid)
        # ------------------------------------------------------------------ #
        # Build point clouds per-environment
        # ------------------------------------------------------------------ #
        for env_idx in range(args.batch_size):
            current_global_env_id = global_env_id + env_idx
            env_dir = DATASET_DIR / f"env_{current_global_env_id:03d}"
            
            rgb_np   = rgb_batch[env_idx].astype(np.uint8)
            depth_np = depth_batch[env_idx]

            # Save RGB image
            rgb_path = env_dir / "rgb" / f"{pose_idx:04d}.png"
            cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

            # Save depth image (it is already in mm, save as 16-bit PNG)
            depth_mm_uint16 = depth_np.astype(np.uint16)
            depth_path = env_dir / "depth" / f"{pose_idx:04d}.png"
            cv2.imwrite(str(depth_path), depth_mm_uint16)

            # Skip if depth is invalid (all zeros)
            if np.max(depth_np) == 0:
                continue
            K   = K_batch[env_idx].cpu().numpy()
            Ecv = E_batch[env_idx][:3].cpu().numpy()  # (3,4) world→cam (OpenCV)
            H, W, _ = rgb_np.shape
            rgb_o3d   = o3d.geometry.Image(rgb_np)
            depth_o3d = o3d.geometry.Image(depth_np)
            intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, K[0,0], K[1,1], K[0,2], K[1,2])
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=10.0, convert_rgb_to_intensity=False
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
            if not pcd.has_points():
                continue
            # Transform from camera to world
            cam_to_world = np.linalg.inv(np.vstack([Ecv, [0,0,0,1]]))
            pcd.transform(cam_to_world)
            pts_all[env_idx].append(np.asarray(pcd.points))
            rgb_all[env_idx].append(np.asarray(pcd.colors))


    env.close()
    global_env_id += args.batch_size
if K_env0 is not None:
    intrinsic_path = DATASET_DIR / "intrinsic.txt"
    np.savetxt(intrinsic_path, K_env0, fmt="%.8f")
    print(f"Saved intrinsics for env_000 to {intrinsic_path}")

print("Done.")

# MARK: Visualize point cloud for env_000
env0_dir = DATASET_DIR / "env_000"
if env0_dir.exists():
    print("Visualizing point cloud for env_000...")
    if K_env0 is None:
        print("Could not get intrinsics for env_000. Skipping visualization.")
    else:
        rgb_files = sorted((env0_dir / "rgb").glob("*.png"))
        depth_files = sorted((env0_dir / "depth").glob("*.png"))

        if not rgb_files:
            print("No images found for env_000. Skipping visualization.")
        else:
            pcds = []
            img_for_shape = cv2.imread(str(rgb_files[0]))
            H, W = img_for_shape.shape[:2]
            intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, K_env0[0,0], K_env0[1,1], K_env0[0,2], K_env0[1,2])

            for i, (rgb_path, depth_path) in enumerate(zip(rgb_files, depth_files)):
                rgb_bgr = cv2.imread(str(rgb_path))
                if rgb_bgr is None:
                    continue
                rgb_np = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                
                depth_np = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if depth_np is None or np.max(depth_np) == 0:
                    continue
                
                rgb_o3d = o3d.geometry.Image(rgb_np)
                depth_o3d = o3d.geometry.Image(depth_np)
                
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    rgb_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=10.0, convert_rgb_to_intensity=False
                )
                
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
                if not pcd.has_points():
                    continue

                cam_to_world = all_poses[i]
                pcd.transform(cam_to_world)
                pcds.append(pcd)
            
            if pcds:
                # Aggregate point clouds
                aggregated_pcd = o3d.geometry.PointCloud()
                for pcd in pcds:
                    aggregated_pcd += pcd
                
                # Downsample
                aggregated_pcd = aggregated_pcd.voxel_down_sample(0.02)
                
                print("[VIS] Launching Open3D visualizer for env_000 (press 'q' to exit). XYZ axes are RGB (Red=X, Green=Y, Blue=Z).")
                axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
                o3d.visualization.draw_geometries([aggregated_pcd, axes], window_name="Env_000 Aggregated PointCloud")