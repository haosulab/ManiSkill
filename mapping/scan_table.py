import os, time, argparse
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(SCRIPT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)
rgb_frames = []

INTRINSIC_TXT = os.path.join(IMG_DIR, "intrinsic.txt")
PLY_PATH      = os.path.join(IMG_DIR, "point_cloud.ply")
HTML_PATH     = os.path.join(IMG_DIR, "point_cloud.html")

def frustum_lines(K: np.ndarray, E_cv: np.ndarray, W: int, H: int, depth=0.25):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    pix = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    dirs = np.stack([(pix[:,0]-cx)/fx,
                     (pix[:,1]-cy)/fy,
                     np.ones(4)], axis=1)
    pts_cam = np.vstack([np.zeros(3), dirs * depth])
    homo    = np.hstack([pts_cam, np.ones((5,1))])

    cam_to_world = np.linalg.inv(np.vstack([E_cv, [0,0,0,1]]))

    return (cam_to_world @ homo.T).T[:, :3]

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

poses = []
pts_all, rgb_all = [], []
frusta = []
calib_written = False

def capture():
    global calib_written
    obs = env.get_obs()
    d = obs["sensor_data"]["moving_camera"]
    p = obs["sensor_param"]["moving_camera"]

    rgb_np   = np.squeeze(d["rgb"].cpu().numpy()).astype(np.uint8)
    rgb_frames.append(rgb_np)
    depth_np = np.squeeze(d["depth"].cpu().numpy())
    K        = np.squeeze(p["intrinsic_cv"].cpu().numpy())
    E_cv     = np.squeeze(p["extrinsic_cv"].cpu().numpy())

    H, W, _ = rgb_np.shape

    if args.frustum:
        frusta.append(frustum_lines(K, E_cv, W, H))

    if not calib_written:
        np.savetxt(INTRINSIC_TXT, K, fmt="%.8f")
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
    poses.append(cam_to_world[:3])
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
o3d.io.write_point_cloud(PLY_PATH, pcd)
print(f"[PLY] wrote {PLY_PATH}")

POSES_TXT = os.path.join(IMG_DIR, "poses.txt")
with open(POSES_TXT, "w") as f:
    for T in poses:
        f.write(" ".join(f"{v:.12e}" for v in T.flatten()) + "\n")
print(f"[POSES] wrote {POSES_TXT}")

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

if args.frustum:
    print("[INFO] Adding frustum visualizations to the plot.")
    for F in frusta:
        x_lines, y_lines, z_lines = [], [], []
        far_plane_order = [1, 2, 3, 4, 1]
        x_lines.extend(F[far_plane_order, 0].tolist() + [None])
        y_lines.extend(F[far_plane_order, 1].tolist() + [None])
        z_lines.extend(F[far_plane_order, 2].tolist() + [None])

        for i in range(1, 5):
            apex_order = [0, i]
            x_lines.extend(F[apex_order, 0].tolist() + [None])
            y_lines.extend(F[apex_order, 1].tolist() + [None])
            z_lines.extend(F[apex_order, 2].tolist() + [None])

        fig.add_trace(go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode="lines",
            line=dict(width=2, color="royalblue"),
            showlegend=False))

fig.update_layout(scene=dict(aspectmode="data"),
                  margin=dict(l=0,r=0,b=0,t=0))
pyo.plot(fig, filename=HTML_PATH, auto_open=False)
print(f"[HTML] wrote {HTML_PATH}")

if args.viewer:
    print("[INFO] Viewer enabled. Keeping window open for up to 120 seconds.")
    start = time.time()
    while not viewer.window.should_close and time.time() - start < 120:
        viewer.render()
    viewer.close()

VIDEO_PATH = os.path.join(IMG_DIR, "capture_rgb.mp4")
if rgb_frames:
    # make sure all frames are identical size
    assert len({img.shape for img in rgb_frames}) == 1, "inconsistent frame sizes"
    imageio.mimsave(VIDEO_PATH, rgb_frames, fps=10)
    print(f"[VIDEO] wrote {VIDEO_PATH}")
else:
    print("[WARN] no RGB frames collected?")


env.close()
print("Done.")