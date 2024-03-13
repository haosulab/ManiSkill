import numpy as np
import open3d as o3d
import tqdm
import transforms3d
from pymp.robot import RobotWrapper
from pymp.utils import toSE3
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation


def norm_vec(x, eps=1e-6):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.where(norm < eps, 0, x / np.maximum(norm, eps))


def rotate_transform(R):
    out = np.zeros(R.shape[:-2] + (4, 4))
    out[..., :3, :3] = R
    out[..., 3, 3] = 1
    return out


def compute_antipodal_contact_points(points, normals, max_width, score_thresh):
    dist = cdist(points, points)

    # Distance between two contact points should be larger than gripper width.
    mask1 = dist <= max_width

    # [n, n, 3], direction between two surface points
    direction = norm_vec(points[None, :] - points[:, None])
    # [n, n]
    cos_angle = np.squeeze(direction @ normals[..., None], -1)

    # Heuristic from S4G
    score = np.abs(cos_angle * cos_angle.T)
    mask2 = score >= score_thresh
    # print(score[0:5, 0:5])

    row, col = np.nonzero(np.triu(np.logical_and(mask1, mask2), k=1))
    return row, col, score[row, col]


def initialize_grasp_poses(points, row, col):
    # Assume the grasp frame is approaching (x), closing (y), ortho (z).
    # The origin is the center of two contact points.
    # Please convert to the gripper you use.

    # The closing vector is segment between two contact points
    displacement = points[col] - points[row]
    closing_vec = norm_vec(displacement)  # [m, 3]
    # Approaching and orthogonal vectors should be searched later.
    U, _, _ = np.linalg.svd(closing_vec[..., None])  # [m, 3, 3]
    approaching_vec = U[..., 1]  # [m, 3]
    assert np.all(np.einsum("nd,nd->n", approaching_vec, closing_vec) <= 1e-6)
    center = (points[col] + points[row]) * 0.5

    grasp_frames = np.tile(np.eye(4), [len(row), 1, 1])
    grasp_frames[:, 0:3, 0] = approaching_vec
    grasp_frames[:, 0:3, 1] = closing_vec
    grasp_frames[:, 0:3, 2] = np.cross(approaching_vec, closing_vec)
    grasp_frames[:, 0:3, 3] = center
    return grasp_frames, np.linalg.norm(displacement, axis=-1) * 0.5


def augment_grasp_poses(grasp_poses, angles):
    Rs = Rotation.from_euler("y", angles).as_matrix()  # [A, 3, 3]
    Ts = rotate_transform(Rs)  # [A, 4, 4]
    out = np.einsum("nij,mjk->nmik", grasp_poses, Ts)
    return out


def compute_grasp_poses(
    mesh_path,
    n_pts,
    gripper_urdf,
    gripper_srdf,
    tcp_link_name,
    mesh_scale=1.0,
    gripper_width=0.08,
    open_ratio=0.95,
    score_thresh=0.97,
    n_angles=36,
    octree_resolution=0.001,
    open_widths=None,
    add_ground=False,
    seed=0,
    vis=False,
):
    # Load shape
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    mesh.scale(mesh_scale, center=(0, 0, 0))
    min_bound = mesh.get_min_bound()
    # Open3d interpolates normals for sampled points.
    pcd = mesh.sample_points_uniformly(number_of_points=n_pts)
    points, normals = np.array(pcd.points), np.array(pcd.normals)

    # 1. Generate contact point pairs
    row, col, score = compute_antipodal_contact_points(
        points, normals, gripper_width * open_ratio, score_thresh
    )
    print("#contact points", len(row))

    # 2. Build initial grasp poses
    grasp_poses, closing_dists = initialize_grasp_poses(points, row, col)

    # 3. Search feasible grasp poses
    angles = np.linspace(0, 2 * np.pi, n_angles)
    grasp_poses = augment_grasp_poses(grasp_poses, angles)
    print(grasp_poses.shape, closing_dists.shape)
    grasp_poses = np.reshape(grasp_poses, [-1, 4, 4])
    closing_dists = np.repeat(closing_dists, n_angles, axis=0)

    # Load gripper urdf
    gripper = RobotWrapper.loadFromURDF(str(gripper_urdf), floating=True)
    gripper.initCollisionPairs()
    gripper.removeCollisionPairsFromSRDF(str(gripper_srdf))
    gripper.addOctree(points, octree_resolution)

    if add_ground:
        gripper.addBox(
            [1, 1, octree_resolution],
            toSE3([0, 0, min_bound[2] - octree_resolution * 2]),
            name="ground",
        )

    if vis:
        gripper.addMeshVisual(np.array(mesh.vertices), np.array(mesh.triangles))

    # TCP frame
    tcp_frame = gripper.get_frame(tcp_link_name)
    root_to_tcp = np.array(tcp_frame.placement.inverse())

    # 4. Search
    valid_grasp_poses = []
    valid_grasp_qpos = []
    pbar = tqdm.tqdm(total=len(grasp_poses))
    if open_widths is None:
        open_widths = [0.01, 0.02, 0.03, 0.04]

    for grasp_pose, closing_dist in zip(grasp_poses, closing_dists):
        root_pose = grasp_pose @ root_to_tcp
        pos = root_pose[:3, 3]
        quat = Rotation.from_matrix(root_pose[:3, :3]).as_quat()
        for w in open_widths:
            if w < closing_dist:
                continue
            grasp_qpos = np.hstack([pos, quat, w, w])
            if gripper.isCollisionFree(grasp_qpos):
                valid_grasp_qpos.append(grasp_qpos)
                # sapien format
                pq = np.hstack(
                    [
                        grasp_pose[:3, 3],
                        transforms3d.quaternions.mat2quat(grasp_pose[:3, :3]),
                    ]
                )
                valid_grasp_poses.append(np.hstack([pq, w]))
                break
        pbar.update()

    valid_grasp_poses = np.array(valid_grasp_poses)
    valid_grasp_qpos = np.array(valid_grasp_qpos)
    print("#valid grasp poses", len(valid_grasp_poses))

    if len(valid_grasp_poses) > 0 and vis:
        gripper.initMeshcatDisplay(None)
        gripper.play(valid_grasp_qpos, dt=0.5)

    return valid_grasp_poses


def main():
    from mani_skill import DESCRIPTION_DIR
    from mani_skill.envs.pick_and_place.pick_single import YCB_DIR

    # Load shape
    # mesh_path = str(YCB_DIR / "models/002_master_chef_can/convex.obj")
    # mesh_path = str(YCB_DIR / "models/065-h_cups/convex.obj")
    mesh_path = str(YCB_DIR / "models/072-a_toy_airplane/convex.obj")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    # Open3d interpolates normals for sampled points.
    pcd = mesh.sample_points_uniformly(number_of_points=2048, seed=0)
    points, normals = np.array(pcd.points), np.array(pcd.normals)

    # vis_pcd = o3d.geometry.PointCloud()
    # vis_pcd.points = o3d.utility.Vector3dVector(points)
    # vis_pcd.normals = o3d.utility.Vector3dVector(normals)
    # o3d.visualization.draw_geometries([vis_pcd])

    # 1. Generate contact point pairs
    row, col, score = compute_antipodal_contact_points(
        points, normals, 0.08 * 0.95, 0.97
    )
    print("#contact points", len(row))

    # 2. Build initial grasp poses
    grasp_poses, closing_dists = initialize_grasp_poses(points, row, col)

    # 3. Search feasible grasp poses
    n_angles = 36
    angles = np.linspace(0, 2 * np.pi, n_angles)
    grasp_poses = augment_grasp_poses(grasp_poses, angles)
    print(grasp_poses.shape)
    grasp_poses = np.reshape(grasp_poses, [-1, 4, 4])
    closing_dists = np.repeat(closing_dists, n_angles, axis=0)

    gripper_urdf = str(DESCRIPTION_DIR / "panda_v2_gripper.urdf")
    gripper_srdf = str(DESCRIPTION_DIR / "panda_v2.srdf")
    gripper = RobotWrapper.loadFromURDF(gripper_urdf, floating=True)
    # print(gripper.nq, gripper.nv)
    # print(gripper.model.names)
    # print(gripper.model.frames)

    gripper.initCollisionPairs()
    gripper.removeCollisionPairsFromSRDF(gripper_srdf)
    gripper.addOctree(points, 0.001)
    gripper.addMeshVisual(np.array(mesh.vertices), np.array(mesh.triangles))
    # # Add a ground
    # gripper.addBox([1, 1, 0.005], toSE3([0, 0, -0.005]), name="ground")

    # TCP frame
    tcp_frame = gripper.get_frame("panda_hand_tcp")
    root_to_tcp = np.array(tcp_frame.placement.inverse())

    # 4. Search
    valid_grasp_poses = []
    pbar = tqdm.tqdm(total=len(grasp_poses))
    w_step = 0.005
    widths = np.arange(0.01, 0.04 + w_step * 0.5, w_step)

    for grasp_pose, closing_dist in zip(grasp_poses, closing_dists):
        root_pose = grasp_pose @ root_to_tcp
        pos = root_pose[:3, 3]
        quat = Rotation.from_matrix(root_pose[:3, :3]).as_quat()
        for w in widths:
            if w < closing_dist:
                continue
            grasp_qpos = np.hstack([pos, quat, w, w])
            if gripper.isCollisionFree(grasp_qpos):
                valid_grasp_poses.append(grasp_qpos)
                break
        pbar.update()

    valid_grasp_poses = np.array(valid_grasp_poses)
    print("#valid grasp poses", len(valid_grasp_poses))

    if len(valid_grasp_poses) == 0:
        exit(0)

    # Visualize
    gripper.initMeshcatDisplay(None)
    gripper.play(np.array(valid_grasp_poses).T, dt=0.5)


if __name__ == "__main__":
    main()
