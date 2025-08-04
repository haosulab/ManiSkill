import os.path as osp
from dataclasses import dataclass
from typing import List, Optional

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import nvdiffrast.torch as dr
import sapien
import torch
import torch.nn as nn
import transforms3d
import trimesh
import tyro
from tqdm import tqdm
from utils import utils_3d
from utils.nvdiffrast_utils import K_to_projection, transform_pos
from utils.utils_3d import se3_exp_map, se3_log_map

import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv


class NVDiffrastRenderer:
    def __init__(self, width, height):
        self.H, self.W = height, width
        self.resolution = (height, width)
        blender2opencv = (
            torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            .float()
            .cuda()
        )
        self.opencv2blender = torch.inverse(blender2opencv)
        self.glctx = dr.RasterizeCudaContext()

    def render_mask(self, verts, faces, K, object_pose, anti_aliasing=True):
        proj = K_to_projection(K, self.H, self.W)

        pose = self.opencv2blender @ object_pose

        pos_clip = transform_pos(proj @ pose, verts)

        rast_out, _ = dr.rasterize(
            self.glctx, pos_clip, faces, resolution=self.resolution
        )
        if anti_aliasing:
            vtx_color = torch.ones(verts.shape, dtype=torch.float, device=verts.device)
            color, _ = dr.interpolate(vtx_color[None, ...], rast_out, faces)
            color = dr.antialias(color, rast_out, pos_clip, faces)
            mask = color[0, :, :, 0]
        else:
            mask = rast_out[0, :, :, 2] > 0
        mask = torch.flip(mask, dims=[0])
        return mask

    def batch_render_mask(self, verts, faces, K, anti_aliasing=True):
        proj = K_to_projection(K, self.H, self.W)

        pose = self.opencv2blender

        pos_clip = transform_pos(proj @ pose, verts)

        rast_out, _ = dr.rasterize(
            self.glctx, pos_clip, faces, resolution=self.resolution
        )
        if anti_aliasing:
            vtx_color = torch.ones(verts.shape, dtype=torch.float, device=verts.device)
            color, _ = dr.interpolate(vtx_color[None, ...], rast_out, faces)
            color = dr.antialias(color, rast_out, pos_clip, faces)
            mask = color[0, :, :, 0]
        else:
            mask = rast_out[0, :, :, 2] > 0
        mask = torch.flip(mask, dims=[0])
        return mask


@dataclass
class RBSolverConfig:
    camera_height: int
    camera_width: int
    robot_masks: torch.Tensor
    link_poses_dataset: torch.Tensor
    mesh_paths: List[str]
    initial_extrinsic_guess: torch.Tensor


class RBSolver(nn.Module):
    """Rendering based solver for inverse rendering based extrinsic prediction"""

    def __init__(self, cfg: RBSolverConfig):
        super().__init__()
        self.cfg = cfg
        mesh_paths = self.cfg.mesh_paths
        for link_idx, mesh_path in enumerate(mesh_paths):
            mesh = trimesh.load(osp.expanduser(mesh_path), force="mesh")
            vertices = torch.from_numpy(mesh.vertices).float()
            faces = torch.from_numpy(mesh.faces).int()
            self.register_buffer(f"vertices_{link_idx}", vertices)
            self.register_buffer(f"faces_{link_idx}", faces)
        self.nlinks = len(mesh_paths)
        # camera parameters
        init_Tc_c2b = self.cfg.initial_extrinsic_guess
        init_dof = se3_log_map(
            torch.as_tensor(init_Tc_c2b, dtype=torch.float32)[None].permute(0, 2, 1),
            eps=1e-5,
            backend="opencv",
        )[0]
        self.dof = nn.Parameter(init_dof, requires_grad=True)
        # setup renderer
        self.H, self.W = self.cfg.camera_height, self.cfg.camera_width
        self.renderer = NVDiffrastRenderer(self.H, self.W)

        self.register_buffer(f"history_ops", torch.zeros(10000, 6))

    def forward(self, data):
        put_id = (self.history_ops == 0).all(dim=1).nonzero()[0, 0].item()
        self.history_ops[put_id] = self.dof.detach()
        Tc_c2b = se3_exp_map(self.dof[None]).permute(0, 2, 1)[0]
        losses = []
        all_frame_all_link_si = []
        masks_ref = data["mask"]
        link_poses = data["link_poses"]
        assert link_poses.shape[0] == masks_ref.shape[0]
        assert link_poses.shape[1:] == (self.nlinks, 4, 4)
        assert masks_ref.shape[1:] == (self.H, self.W)
        intrinsic = data["intrinsic"]

        batch_size = masks_ref.shape[0]
        for bid in range(batch_size):
            all_link_si = []
            for link_idx in range(self.nlinks):
                Tc_c2l = Tc_c2b @ link_poses[bid, link_idx]
                verts, faces = getattr(self, f"vertices_{link_idx}"), getattr(
                    self, f"faces_{link_idx}"
                )
                si = self.renderer.render_mask(
                    verts, faces, K=intrinsic, object_pose=Tc_c2l
                )
                all_link_si.append(si)
            all_link_si = torch.stack(all_link_si).sum(0).clamp(max=1)
            all_frame_all_link_si.append(all_link_si)

            loss = torch.sum((all_link_si - masks_ref[bid].float()) ** 2)
            losses.append(loss)
        loss = torch.stack(losses).mean()
        all_frame_all_link_si = torch.stack(all_frame_all_link_si)
        output = {
            "rendered_masks": all_frame_all_link_si,
            "ref_masks": masks_ref,
            "error_maps": (all_frame_all_link_si - masks_ref.float()).abs(),
        }
        # metrics
        output = dict()
        if "gt_camera_pose" in data:
            gt_Tc_c2b = data["gt_camera_pose"]
            if not torch.allclose(gt_Tc_c2b, torch.eye(4).to(gt_Tc_c2b.device)):
                gt_dof6 = utils_3d.se3_log_map(
                    gt_Tc_c2b[None].permute(0, 2, 1), backend="opencv"
                )[0]
                trans_err = ((gt_dof6[:3] - self.dof[:3]) * 100).abs()
                rot_err = (gt_dof6[3:] - self.dof[3:]).abs().max() / np.pi * 180

                metrics = {
                    "err_x": trans_err[0].item(),
                    "err_y": trans_err[1].item(),
                    "err_z": trans_err[2].item(),
                    "err_trans": trans_err.norm().item(),
                    "err_rot": rot_err.item(),
                }
                output["metrics"] = metrics
        output["mask_loss"] = loss
        return output

    def get_predicted_extrinsic(self):
        return utils_3d.se3_exp_map(self.dof[None].detach()).permute(0, 2, 1)[0]


def optimize(
    camera_intrinsic: torch.Tensor,
    robot_masks: torch.Tensor,
    link_poses_dataset: torch.Tensor,
    initial_extrinsic_guess: torch.Tensor,
    mesh_paths: List[str],
    camera_width: int,
    camera_height: int,
    iterations: int = 200,
    gt_camera_pose: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
):
    """
    Optimizes an initial guess of a camera extrinsic using the camera intrinsic matrix, a dataset of robot masks, link poses relative to the robot base frame, and paths to the mesh files of each of the link poses.

    Parameters:

        camera_intrinsic (torch.Tensor, shape (3, 3)): Camera intrinsic matrix
        robot_masks (torch.Tensor, shape (N, H, W)): Robot segmentation masks
        link_poses_dataset (torch.Tensor, shape (N, L, 4, 4)): Link poses relative to the robot base frame, where N is the number of samples, L is the number of links
        initial_extrinsic_guess (torch.Tensor, shape (4, 4)): Initial guess of the camera extrinsic
        mesh_paths (List[str]): List of paths to the mesh files of each of the L links
        camera_width (int): Camera width
        camera_height (int): Camera height
        iterations (int): Number of optimization iterations
        batch_size (int): Default is None meaning whole batch optimization. Otherwise this specifies the number of samples to process in each batch.
        gt_camera_pose (torch.Tensor, shape (4, 4)): Default is None. If a ground truth camera pose is provided the optimization function will compute error metrics relative to the ground truth camera pose.
    """
    device = robot_masks.device
    cfg = RBSolverConfig(
        camera_width=camera_width,
        camera_height=camera_height,
        robot_masks=robot_masks,
        link_poses_dataset=link_poses_dataset,
        mesh_paths=mesh_paths,
        initial_extrinsic_guess=initial_extrinsic_guess,
    )
    solver = RBSolver(cfg)
    solver = solver.to(device)
    optimizer = torch.optim.Adam(solver.parameters(), lr=3e-3)
    best_predicted_extrinsic = initial_extrinsic_guess.clone()
    best_loss = float("inf")
    pbar = tqdm(range(iterations))
    dataset = dict(
        intrinsic=camera_intrinsic,
        link_poses=link_poses_dataset,
        mask=robot_masks,
    )
    if gt_camera_pose is not None:
        dataset["gt_camera_pose"] = gt_camera_pose
    for i in pbar:
        if batch_size is None:
            batch = dataset
        else:
            bid = torch.randperm(len(dataset["mask"]))[:batch_size]
            batch = {k: v[bid] for k, v in dataset.items()}
        output = solver(batch)
        optimizer.zero_grad()
        output["mask_loss"].backward()
        optimizer.step()
        loss_value = output["mask_loss"].item()
        pbar.set_description(f"Loss: {loss_value:.4f}")
        if loss_value < best_loss:
            best_loss = loss_value
            best_predicted_extrinsic = solver.get_predicted_extrinsic()
            if loss_value < 1000:
                break
        if "metrics" in output:
            pbar.set_postfix(output["metrics"])
    return best_predicted_extrinsic


@dataclass
class Args:
    image_path: Optional[str] = None
    """path to a folder containing all the images from the same fixed camera of a real robot at different joint configurations"""
    env_id: Optional[str] = None
    """the simulated environment ID if you want to test easy hec on a sim environment. With the sim environment it will use ground truth segmentation masks for optimization"""
    batch_size: int = 16
    train_steps: int = 100
    seed: int = 0


def main(args: Args):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.env_id:
        num_synthetic_samples = 20
        env = gym.make(args.env_id, shader_dir="rt-fast")
        env.reset()
        base_env: BaseEnv = env.unwrapped
        render_camera = base_env._human_render_cameras["render_camera"]
        intrinsic = render_camera.camera.get_intrinsic_matrix()[0].cpu().numpy()

        # gets ground truth camera pose
        camera_pose = np.eye(4)
        extrinsic = render_camera.camera.get_extrinsic_matrix()[0].cpu().numpy()
        extrinsic[:3, :3] = extrinsic[:3, :3]
        camera_pose[:3, :4] = extrinsic
        camera_height, camera_width = (
            render_camera.config.height,
            render_camera.config.width,
        )

        # find all visible links
        visible_links = [
            x
            for x in base_env.agent.robot.links
            if x._objs[0].entity.find_component_by_type(
                sapien.render.RenderBodyComponent
            )
            is not None
        ]

        # generate synthetic rgb images, segmentation images, and link poses of the robot in the scene.
        images = []
        segmentation_masks = []
        link_poses_dataset = []
        init_qpos = base_env.agent.robot.get_qpos().cpu().numpy()
        for i in range(num_synthetic_samples):
            noise = np.random.randn(init_qpos.shape[1]) * 0.5
            noise[-2:] = 0
            base_env.agent.robot.set_qpos(init_qpos.copy() + noise)
            base_env.scene.update_render()
            render_camera.capture()
            data = render_camera.get_obs(rgb=True, segmentation=True)
            images.append(data["rgb"].clone())
            segmentation_masks.append(data["segmentation"][..., 0].clone())

            link_poses = []
            for link in visible_links:
                # NOTE (stao): assumes there is only 1 render shape and 1 render body component per link
                # A fix would be to treat each render shape as a separate "link" next
                link_visual_mesh_pose = (
                    link.pose.sp
                    * link._objs[0]
                    .entity.find_component_by_type(sapien.render.RenderBodyComponent)
                    .render_shapes[0]
                    .local_pose
                )
                link_poses.append(link_visual_mesh_pose.to_transformation_matrix())
            link_poses_dataset.append(np.stack(link_poses))
        link_poses_dataset = np.stack(link_poses_dataset)

        segmentation_images = []
        robot_masks = []
        for i in range(len(images)):
            images[i] = images[i][0].cpu().numpy()
            segmentation_masks[i] = segmentation_masks[i][0].cpu().numpy()
        for i in range(len(segmentation_masks)):
            segmentation_image = images[i].copy()
            segment_ids = []
            for link in base_env.agent.robot.links:
                segment_ids.append(link.per_scene_id[0])
            robot_mask = np.isin(segmentation_masks[i], segment_ids)
            segmentation_image[robot_mask] //= 4
            robot_masks.append(robot_mask)
            segmentation_images.append(segmentation_image)
        robot_masks = np.stack(robot_masks)

        mesh_paths = []
        for link in visible_links:
            # assumes there is only one render body component
            rb = link._objs[0].entity.find_component_by_type(
                sapien.render.RenderBodyComponent
            )
            if rb is None:
                continue
            for render_shape in rb.render_shapes:
                mesh_filename = render_shape.filename
                mesh_paths.append(mesh_filename)

        # generate an initial guess around the ground truth pose. This is off by +-10cm
        initial_extrinsic_guess = camera_pose.copy()
        initial_extrinsic_guess[:3, 3] -= 0.3
        initial_extrinsic_guess[1, 3] += 0.3
        # Add random rotation perturbation to initial guess
        random_axis = np.array([-1, 0.3, 1])
        random_axis = random_axis / np.linalg.norm(
            random_axis
        )  # normalize to unit vector
        random_angle = np.deg2rad(
            30
        )  # Random angle between -0.1 and 0.1 radians (~5.7 degrees)
        rotation_matrix = transforms3d.axangles.axangle2mat(random_axis, random_angle)
        initial_extrinsic_guess[:3, :3] = (
            rotation_matrix @ initial_extrinsic_guess[:3, :3]
        )
        initial_extrinsic_guess = (
            torch.tensor(initial_extrinsic_guess).float().to(device)
        )
        camera_pose = torch.tensor(camera_pose).float().to(device)
        robot_masks = torch.from_numpy(robot_masks).float().to(device)
        link_poses_dataset = torch.from_numpy(link_poses_dataset).float().to(device)
        intrinsic = torch.from_numpy(intrinsic).float().to(device)
        predicted_camera_extrinsic = optimize(
            camera_intrinsic=intrinsic,
            robot_masks=robot_masks,
            link_poses_dataset=link_poses_dataset,
            initial_extrinsic_guess=initial_extrinsic_guess,
            mesh_paths=mesh_paths,
            camera_width=camera_width,
            camera_height=camera_height,
            gt_camera_pose=camera_pose,
            iterations=args.train_steps,
        )
    else:
        if not args.image_path:
            raise ValueError("Must provide either env_id or image_path")

        # # Get all image files in the folder
        # image_files = []
        # for ext in ['*.jpg', '*.jpeg', '*.png']:
        #     image_files.extend(glob.glob(os.path.join(args.image_path, ext)))

        # if len(image_files) == 0:
        #     raise ValueError(f"No image files found in {args.image_path}")

        # print(f"Found {len(image_files)} images in {args.image_path}")

        # # Load camera parameters
        # intrinsic = np.loadtxt(os.path.join(args.image_path, "K.txt"))
        # camera_pose = np.eye(4)
        # camera_pose = torch.from_numpy(camera_pose).float().cuda()
        # camera_pose.requires_grad = True

        # # Get image dimensions from first image
        # img = cv2.imread(image_files[0])
        # camera_height, camera_width = img.shape[:2]

    ### visulization code ###
    renderer = NVDiffrastRenderer(camera_width, camera_height)
    for i in range(len(images)):

        def get_mask_from_camera_pose(camera_pose):
            mask = torch.zeros((camera_height, camera_width), device=base_env.device)
            for j, link_pose in enumerate(link_poses_dataset[i]):
                mesh_path = mesh_paths[j]
                mesh = trimesh.load_mesh(mesh_path)
                vertices = mesh.vertices.copy()
                link_mask = renderer.render_mask(
                    torch.from_numpy(vertices).float().to(device),
                    torch.from_numpy(mesh.faces).int().to(device),
                    intrinsic,
                    camera_pose @ link_pose,
                )
                link_mask = link_mask.detach()
                mask[link_mask > 0] = 1
            return mask

        initial_guess_mask = get_mask_from_camera_pose(initial_extrinsic_guess)
        predicted_mask = get_mask_from_camera_pose(predicted_camera_extrinsic)
        initial_guess_mask = initial_guess_mask.cpu().numpy()
        predicted_mask = predicted_mask.cpu().numpy()
        overlaid_image_initial_guess = images[i].copy()
        overlaid_image_predicted = images[i].copy()
        overlaid_image_initial_guess[initial_guess_mask > 0] = (
            overlaid_image_initial_guess[initial_guess_mask > 0] // 4
        )
        overlaid_image_predicted[predicted_mask > 0] = (
            overlaid_image_predicted[predicted_mask > 0] // 4
        )

        plt.figure(figsize=(21, 7))

        plt.subplot(1, 3, 1)
        plt.imshow(overlaid_image_initial_guess)
        plt.title("Original Extrinsic Guess")

        plt.subplot(1, 3, 2)
        plt.imshow(overlaid_image_predicted)
        plt.title("Predicted Extrinsic")

        plt.subplot(1, 3, 3)
        plt.imshow(segmentation_images[i], cmap="gray")
        plt.title("Robot Segmentation Mask")

        plt.show()


if __name__ == "__main__":
    main(tyro.cli(Args))
