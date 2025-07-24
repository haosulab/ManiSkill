import os, time, argparse
import numpy as np
import open3d as o3d
import plotly.graph_objs as go
import plotly.offline as pyo
import gymnasium as gym
import imageio.v2 as imageio
import sapien
import mani_skill.envs.tasks.tabletop.table_scan


import torch, torch.nn.functional as F
import open_clip
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# local modules (copied under mapping/)
from mapping_lib.utils import get_visual_features, get_3d_coordinates, transform
from mapping_lib.voxel_hash_table import VoxelHashTable
from mapping_lib.implicit_decoder import ImplicitDecoder

parser = argparse.ArgumentParser(description="Scan an object and generate a point cloud.")

parser.add_argument(
    "--viewer",
    action="store_true",
    help="Enable the SAPIEN viewer for real-time visualization."
)
parser.add_argument(
    "--save",
    dest="save",
    action="store_true",
    help="Save the trained voxel grid & decoder."
)
parser.add_argument(
    "--pca",
    dest="pca",
    action="store_true",
    help="Generate PCA visualization of voxel features."
)
args = parser.parse_args()

# --------------------------------------------------------------------------- #
#  Device / CLIP model                                                        #
# --------------------------------------------------------------------------- #

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
clip_model_name  = "EVA02-L-14"
clip_weights_id  = "merged2b_s4b_b131k"

clip_model, _, _ = open_clip.create_model_and_transforms(
    clip_model_name, pretrained=clip_weights_id
)
clip_model = clip_model.to(DEVICE).eval()

# --------------------------------------------------------------------------- #
#  Grid + Decoder initialization                                              #
# --------------------------------------------------------------------------- #

GRID_RES          = 0.05       # ~5 cm voxels
GRID_LVLS         = 2
GRID_FEAT_DIM     = 64
SCENE_MIN         = (-0.6, -0.8, -0.3)
SCENE_MAX         = (0.2,  0.8,  0.3)

grid = VoxelHashTable(
    resolution=GRID_RES,
    num_levels=GRID_LVLS,
    feature_dim=GRID_FEAT_DIM,
    scene_bound_min=SCENE_MIN,
    scene_bound_max=SCENE_MAX,
    device=DEVICE,
    mode="train",
)

decoder = ImplicitDecoder(
    voxel_feature_dim=GRID_FEAT_DIM * GRID_LVLS,
    hidden_dim=768,
    output_dim=768,
).to(DEVICE)

# Simple optimizer to align grid ↔ CLIP features
OPT_LR   = 1e-3
optimizer = torch.optim.Adam(list(grid.parameters()) + list(decoder.parameters()), lr=OPT_LR)

# Logging helpers
step_counter = 0
loss_history = []

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(SCRIPT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

INTRINSIC_TXT = os.path.join(IMG_DIR, "intrinsic.txt")
PLY_PATH      = os.path.join(IMG_DIR, "point_cloud.ply")
HTML_PATH     = os.path.join(IMG_DIR, "point_cloud.html")

# Paths to save learned grid / decoder
GRID_PT_PATH    = os.path.join(IMG_DIR, "voxel_grid.pt")
DECODER_PT_PATH = os.path.join(IMG_DIR, "implicit_decoder.pt")

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
    global calib_written, step_counter
    
    obs = env.get_obs()
    d = obs["sensor_data"]["moving_camera"]
    p = obs["sensor_param"]["moving_camera"]

    rgb_np   = np.squeeze(d["rgb"].cpu().numpy()).astype(np.uint8)
    depth_np = np.squeeze(d["depth"].cpu().numpy())
    K        = np.squeeze(p["intrinsic_cv"].cpu().numpy())
    E_cv     = np.squeeze(p["extrinsic_cv"].cpu().numpy())

    H, W, _ = rgb_np.shape

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

    # ----------------------------------------------------------------------- #
    #  Dense CLIP feature ↔ 3D voxel alignment                                #
    # ----------------------------------------------------------------------- #

    # 1) Prepare RGB tensor for CLIP vision encoder
    img_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0  # (3,H,W)
    img_tensor = transform(img_tensor).unsqueeze(0).to(DEVICE)             # (1,3,224,224)

    with torch.no_grad():
        vis_feat = get_visual_features(clip_model, img_tensor)  # (1,C,16,16)

    # 2) Depth processing – down-sample to 16×16 and metres
    depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).float() / 1000.0  # (1,1,H,W)
    depth_t = F.interpolate(depth_t, (16, 16), mode="nearest-exact").squeeze(1)      # (1,16,16)

    # 3) Camera extrinsic → torch
    extrinsic_t = torch.from_numpy(E_cv).unsqueeze(0).unsqueeze(0).float()           # (1,1,3,4)

    # 4) 3D world coordinates of each patch center
    coords_world, _ = get_3d_coordinates(
        depth_t.to(DEVICE), extrinsic_t.to(DEVICE),
        fx=154.15475464, fy=154.15475464, cx=112, cy=112,
    )                                                   # (1,3,16,16)

    # 5) Flatten
    B, C_, Hf, Wf = vis_feat.shape  # B=1, C_=768, Hf=Wf=16
    N = Hf * Wf
    feats_flat   = vis_feat.permute(0, 2, 3, 1).reshape(-1, C_)           # (N,768)
    coords_flat  = coords_world.permute(0, 2, 3, 1).reshape(-1, 3)         # (N,3)

    # 6) Filter valid depth (>0.1 m)
    depth_flat   = depth_t.view(-1)
    valid_mask   = depth_flat > 0.0
    if valid_mask.sum() == 0:
        return

    coords_valid = coords_flat[valid_mask]
    feats_valid  = feats_flat[valid_mask]

    # 6-b) keep only pts inside SCENE bounds
    in_x = (coords_valid[:,0] >= SCENE_MIN[0]) & (coords_valid[:,0] <= SCENE_MAX[0])
    in_y = (coords_valid[:,1] >= SCENE_MIN[1]) & (coords_valid[:,1] <= SCENE_MAX[1])
    in_z = (coords_valid[:,2] >= SCENE_MIN[2]) & (coords_valid[:,2] <= SCENE_MAX[2])
    in_bounds = in_x & in_y & in_z

    if in_bounds.sum() == 0:
        return  # nothing to train on this capture

    coords_valid = coords_valid[in_bounds].to(DEVICE)
    feats_valid  = feats_valid[in_bounds].to(DEVICE)

    # 7) Query voxel grid, decode, and compute alignment loss
    voxel_feat   = grid.query_voxel_feature(coords_valid)         # (M, grid_feat)
    pred_feat    = decoder(voxel_feat)

    cos_sim      = F.cosine_similarity(pred_feat, feats_valid, dim=-1)
    loss         = 1.0 - cos_sim.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    step_counter += 1
    loss_history.append(loss.item())
    if step_counter % 10 == 0:
        avg_loss = sum(loss_history[-10:]) / 10
        print(f"[Train] step={step_counter:04d}  loss={avg_loss:.4f}")

def deg(*angles): return np.deg2rad(angles)
motions = [
    np.concatenate([deg(   0, -70,  90,  30, 50,  0), [0.04]]),
    np.concatenate([deg(  20, -70,  90, -30, 50,  0), [0.04]]),
    np.concatenate([deg(  50, -70,  50,  15, 50,  0), [0.04]]),
    np.concatenate([deg( -20, -70,  90, -30, 50,  0), [0.04]]),
]

EPOCHS = 10
for epoch in range(EPOCHS):
    print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
    # reset environment each epoch for variation
    env.reset(seed=epoch)
    hide_robot_visuals(env.unwrapped.agent.robot)

    for action in motions:
        for t in range(100):
            env.step(action)
            capture()

    # final capture at end of epoch
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

fig.update_layout(scene=dict(aspectmode="data"),
                  margin=dict(l=0,r=0,b=0,t=0))
pyo.plot(fig, filename=HTML_PATH, auto_open=False)
print(f"[HTML] wrote {HTML_PATH}")

# --------------------------------------------------------------------------- #
#  Save mappings (grid + decoder) & CLIP features                             #
# --------------------------------------------------------------------------- #

if args.save:
    grid.save_dense(GRID_PT_PATH)
    torch.save(decoder.state_dict(), DECODER_PT_PATH)
    print(f"[SAVE] voxel grid → {GRID_PT_PATH}\n       implicit decoder → {DECODER_PT_PATH}")

# --------------------------------------------------------------------- #
#  PCA visualization of voxel features                                  #
# --------------------------------------------------------------------- #
if args.pca:
    print("[VIS] running PCA on voxel features …")

    # use down-sampled point cloud vertices as probe points
    vertices_np = np.vstack(pts_all).astype(np.float32)

    # (1) voxel down-sample
    ds_size = 0.05  # metre grid
    voxel_idx = np.floor(vertices_np / ds_size).astype(np.int32)
    _, uniq = np.unique(voxel_idx, axis=0, return_index=True)
    vertices_ds = vertices_np[uniq]

    # (2) keep only points within SCENE bounds
    in_x = (vertices_ds[:,0] >= SCENE_MIN[0]) & (vertices_ds[:,0] <= SCENE_MAX[0])
    in_y = (vertices_ds[:,1] >= SCENE_MIN[1]) & (vertices_ds[:,1] <= SCENE_MAX[1])
    in_z = (vertices_ds[:,2] >= SCENE_MIN[2]) & (vertices_ds[:,2] <= SCENE_MAX[2])
    in_bounds = in_x & in_y & in_z
    vertices_vis = vertices_ds[in_bounds]

    if vertices_vis.shape[0] == 0:
        print("[VIS] No vertices inside grid bounds; skipping PCA visualization.")
    else:
        coords_t = torch.from_numpy(vertices_vis).to(DEVICE)

        # (3) query voxel features → decoder
        with torch.no_grad():
            voxel_feat = grid.query_voxel_feature(coords_t)
            feats_t    = decoder(voxel_feat)

        feats_np = feats_t.cpu().numpy()

        pca = PCA(n_components=3)
        feats_pca = pca.fit_transform(feats_np)
        scaler = MinMaxScaler()
        feats_pca_norm = scaler.fit_transform(feats_pca)

        # Save PCA-colored point cloud to HTML for quick inspection
        p_fig = go.Figure()
        p_fig.add_trace(go.Scatter3d(x=vertices_vis[:,0], y=vertices_vis[:,1], z=vertices_vis[:,2],
                                     mode="markers",
                                     marker=dict(size=10,
                                                 color=[f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r,g,b in feats_pca_norm],
                                                 opacity=0.8)))
        p_fig.update_layout(scene=dict(aspectmode="data"),
                            margin=dict(l=0,r=0,b=0,t=0))
        PCA_HTML = os.path.join(IMG_DIR, "voxel_pca.html")
        pyo.plot(p_fig, filename=PCA_HTML, auto_open=False)
        print(f"[VIS] PCA visualization saved to {PCA_HTML}")

if args.viewer:
    print("[INFO] Viewer enabled. Keeping window open for up to 120 seconds.")
    start = time.time()
    while not viewer.window.should_close and time.time() - start < 120:
        viewer.render()
    viewer.close()

env.close()
print("Done.")