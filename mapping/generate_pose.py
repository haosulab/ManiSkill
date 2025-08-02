import os, time, argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import plotly.graph_objs as go
import plotly.offline as pyo
import gymnasium as gym
import imageio.v2 as imageio
import sapien
import mani_skill.envs.tasks.tabletop.table_scan
import mani_skill.envs.tasks.tabletop.table_scan_discrete

parser = argparse.ArgumentParser(description="Scan an object and generate a point cloud.")
parser.add_argument(
    "--frustum",
    action="store_true",
    help="Visualize camera frustums in the output HTML file."
)
parser.add_argument(
    "--viewer",
    action="store_true",
    help="Enable the SAPIEN viewer for real-time visualization."
)
args = parser.parse_args()

SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = SCRIPT_DIR / "images"
IMG_DIR.mkdir(exist_ok=True)
POSE_DIR = SCRIPT_DIR / "poses"
POSE_DIR.mkdir(exist_ok=True)
rgb_frames = []

INTRINSIC_TXT = IMG_DIR / "intrinsic.txt"
PLY_PATH      = IMG_DIR / "point_cloud.ply"
HTML_PATH     = IMG_DIR / "point_cloud.html"

def hide_robot_visuals(robot):
    robot.set_root_pose(sapien.Pose([1e3, 1e3, 1e3]))

render_mode = "human" if args.viewer else "rgb_array"
env = gym.make(
    "TableScan-v0",
    robot_uids="xarm6_robotiq",
    obs_mode="rgbd",
    control_mode="pd_joint_pos",
    reward_mode="none",
    render_mode=render_mode,
)
obs, _ = env.reset(seed=0)

hide_robot_visuals(env.unwrapped.agent.robot)

viewer = None
if args.viewer:
    viewer = env.render()
    if not isinstance(viewer, sapien.utils.Viewer):
        raise TypeError("Env did not return a SAPIEN Viewer.")

pts_all, rgb_all = [], []
frusta = []
calib_written = False
frame_idx = 0

def capture():
    global calib_written, frame_idx
    obs = env.get_obs()
    d = obs["sensor_data"]["moving_camera"]
    p = obs["sensor_param"]["moving_camera"]

    rgb_np   = np.squeeze(d["rgb"].cpu().numpy()).astype(np.uint8)
    rgb_frames.append(rgb_np)
    depth_np = np.squeeze(d["depth"].cpu().numpy())
    K        = np.squeeze(p["intrinsic_cv"].cpu().numpy())
    E_cv     = np.squeeze(p["extrinsic_cv"].cpu().numpy())

    pose_path = POSE_DIR / f"{frame_idx:04d}.npy"
    np.save(pose_path, E_cv)
    print(f"[POSE] wrote {pose_path}")
    frame_idx += 1
    
    H, W, _ = rgb_np.shape


    if not calib_written:
        np.savetxt(str(INTRINSIC_TXT), K, fmt="%.8f")
        calib_written = True

    if np.max(depth_np) == 0:
        print("[WARN] depth is all zeros; skipping frame.")
        return

    rgb_o3d  = o3d.geometry.Image(rgb_np)
    depth_o3d = o3d.geometry.Image(depth_np)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, K[0,0], K[1,1], K[0,2], K[1,2])

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                   rgb_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=10.0, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    if not pcd.has_points():
        print("[WARN] RGB-D → PCD produced 0 points.")
        return

    cam_to_world = np.linalg.inv(np.vstack([E_cv, [0, 0, 0, 1]]))
    pcd.transform(cam_to_world)
    pts_all.append(np.asarray(pcd.points))
    rgb_all.append(np.asarray(pcd.colors))
    print(f"[OK] captured {len(pcd.points)} pts; total {sum(map(len, pts_all))}")

def deg(*angles): return np.deg2rad(angles)
motions = [
    np.concatenate([deg(   0, -70,  90,  30, 50,  0), [0.04]]),
    np.concatenate([deg(  20, -70,  90, -30, 50,  0), [0.04]]),
    np.concatenate([deg(  50, -70,  50,  15, 50,  0), [0.04]]),
    np.concatenate([deg( -20, -70,  90, -30, 50,  0), [0.04]]),
]

for action in motions:
    for t in range(300):
        env.step(action)
        if t % 30 == 0: capture()

capture()

if not pts_all:
    print("\n[ERROR] No points were captured. The script cannot create a point cloud.")
    print("        This might be because the camera did not see any objects.")
    env.close()
    exit()

pcd = o3d.geometry.PointCloud()
pcd.points  = o3d.utility.Vector3dVector(np.vstack(pts_all))
pcd.colors  = o3d.utility.Vector3dVector(np.vstack(rgb_all))
pcd         = pcd.voxel_down_sample(0.02)
o3d.io.write_point_cloud(str(PLY_PATH), pcd)
print(f"[PLY] wrote {PLY_PATH}")

pts  = np.asarray(pcd.points)
cols = (np.asarray(pcd.colors) * 255).astype(np.uint8)
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=pts[:,0], y=pts[:,1], z=pts[:,2],
    mode="markers",
    marker=dict(size=2,
                color=[f"rgb({r},{g},{b})" for r,g,b in cols],
                opacity=0.8),
    name="PCD"))

fig.update_layout(scene=dict(aspectmode="data"),
                  margin=dict(l=0,r=0,b=0,t=0))
pyo.plot(fig, filename=str(HTML_PATH), auto_open=False)
print(f"[HTML] wrote {HTML_PATH}")

if args.viewer:
    print("[INFO] Viewer enabled. Keeping window open for up to 120 seconds.")
    start = time.time()
    while not viewer.window.should_close and time.time() - start < 120:
        viewer.render()
    viewer.close()

env.close()
print("Done.")
