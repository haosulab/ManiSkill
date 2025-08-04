from dataclasses import dataclass
from typing import Optional

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import nvdiffrast.torch as dr
import sapien
import torch
import trimesh
import tyro
from utils.nvdiffrast_utils import K_to_projection, transform_pos

import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv


class NVDiffrastRenderer:
    def __init__(self, image_size):
        """
        image_size: H,W
        """
        # self.
        self.H, self.W = image_size
        self.resolution = image_size
        blender2opencv = (
            torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            .float()
            .cuda()
        )
        self.opencv2blender = torch.inverse(blender2opencv)
        self.glctx = dr.RasterizeCudaContext()

    def render_mask(self, verts, faces, K, object_pose, anti_aliasing=True):
        """
        @param verts: N,3, torch.tensor, float, cuda
        @param faces: M,3, torch.tensor, int32, cuda
        @param K: 3,3 torch.tensor, float ,cuda
        @param object_pose: 4,4 torch.tensor, float, cuda
        @return: mask: 0 to 1, HxW torch.cuda.FloatTensor
        """
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
        """
        @param batch_verts: N,3, torch.tensor, float, cuda
        @param batch_faces: M,3, torch.tensor, int32, cuda
        @param K: 3,3 torch.tensor, float ,cuda
        # @param batch_object_poses: N,4,4 torch.tensor, float, cuda
        @return: mask: 0 to 1, HxW torch.cuda.FloatTensor
        """
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


# def main():
#     pose = np.array([[0.99638397, -0.0846324, 0.00750877, -0.20668708],
#                      [-0.00875172, -0.19013488, -0.9817189, 0.08405855],
#                      [0.0845129, 0.97810328, -0.19018805, 0.77892876],
#                      [0., 0., 0., 1.]]).astype(np.float32)
#     pose = torch.from_numpy(pose).cuda()
#     pose.requires_grad = True
#     mesh = trimesh.load_mesh("data/xarm_description/meshes/xarm7/visual/link0.STL")
#     K = np.loadtxt("data/realsense/20230124_092547/K.txt")
#     H, W = 720, 1280
#     renderer = NVDiffrastRenderer([H, W])
#     mask = renderer.render_mask(torch.from_numpy(mesh.vertices).cuda().float(),
#                                 torch.from_numpy(mesh.faces).cuda().int(),
#                                 torch.from_numpy(K).cuda().float(),
#                                 pose)
#     plt.imshow(mask.detach().cpu())
#     plt.show()


@dataclass
class Args:
    image_path: Optional[str] = None
    """path to a folder containing all the images from the same fixed camera of a real robot at different joint configurations"""
    env_id: Optional[str] = None
    """the simulated environment ID if you want to test easy hec on a sim environment. With the sim environment it will use ground truth segmentation masks for optimization"""


def main(args: Args):

    if args.env_id:
        env = gym.make(args.env_id, shader_dir="rt-fast")
        env.reset()
        base_env: BaseEnv = env.unwrapped
        render_camera = base_env._human_render_cameras["render_camera"]
        intrinsic = render_camera.camera.get_intrinsic_matrix()[0].cpu().numpy()
        camera_pose = np.eye(4)
        extrinsic = render_camera.camera.get_extrinsic_matrix()[0].cpu().numpy()
        extrinsic[:3, :3] = extrinsic[:3, :3]  # @ rot_x @ rot_x
        camera_pose[:3, :4] = extrinsic
        camera_pose = torch.from_numpy(camera_pose).float().cuda()
        camera_pose.requires_grad = True
        camera_height, camera_width = (
            render_camera.config.height,
            render_camera.config.width,
        )

        images = []
        segmentation_masks = []
        for i in range(5):
            base_env.agent.robot.set_qpos(
                base_env.agent.robot.get_qpos()
                + np.random.randn(base_env.agent.robot.get_qpos().shape[0]) * 0.05
            )
            base_env.scene.update_render()
            render_camera.capture()
            data = render_camera.get_obs(rgb=True, segmentation=True)
            images.append(data["rgb"])
            segmentation_masks.append(data["segmentation"])

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
            segmentation_image[robot_mask[..., 0]] //= 4
            robot_masks.append(robot_mask)
            segmentation_images.append(segmentation_image)
        # import ipdb; ipdb.set_trace()
        image = images[-1]
        # for image in images:
        # get the segmentation mask

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

    renderer = NVDiffrastRenderer([camera_height, camera_width])

    mask = torch.zeros((camera_height, camera_width), device=base_env.device)
    for link in base_env.agent.robot.links:
        # assumes there is only one render body component
        rb = link._objs[0].entity.find_component_by_type(
            sapien.render.RenderBodyComponent
        )
        if rb is None:
            continue
        for render_shape in rb.render_shapes:
            mesh_filename = render_shape.filename
            # TODO (stao): this does not handle primitives atm
            mesh = trimesh.load_mesh(mesh_filename)
            mesh_pose = link.pose.sp * render_shape.local_pose

            vertices = mesh.vertices.copy()
            vertices_homo = np.concatenate(
                [vertices, np.ones((vertices.shape[0], 1))], axis=1
            )
            link_pose = mesh_pose.to_transformation_matrix()
            # link_pose += np.random.randn(4, 4) * 0.01
            # transform the mesh for nvdiffrast renderer
            vertices = (vertices_homo @ link_pose.T)[:, :3]
            link_mask = renderer.render_mask(
                torch.from_numpy(vertices).cuda().float(),
                torch.from_numpy(mesh.faces).cuda().int(),
                torch.from_numpy(intrinsic).cuda().float(),
                camera_pose,
            )
            link_mask = link_mask.detach()
            mask[link_mask > 0] = 1

    # shade the original image based on mask
    mask = mask.cpu().numpy()
    image[mask > 0] = image[mask > 0] // 4
    image = image

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Shaded Image")

    plt.subplot(1, 2, 2)
    plt.imshow(segmentation_images[0], cmap="gray")
    plt.title("Segmentation Mask")

    plt.show()


if __name__ == "__main__":
    main(tyro.cli(Args))
