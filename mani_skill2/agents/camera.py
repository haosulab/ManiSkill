import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import sapien.core as sapien

from mani_skill2.utils.sapien_utils import get_entity_by_name, set_actor_visibility

_USE_DLPACK = os.environ.get("MANISKILL2_USE_DLPACK") == "1"
if _USE_DLPACK:
    try:
        import torch
        from torch.utils.dlpack import from_dlpack

        major, minor, patch = torch.__version__.split(".")[:3]
        if not (int(major) >= 1 and int(minor) >= 5):
            _USE_DLPACK = False
    except ImportError:
        _USE_DLPACK = False


def get_texture_naive(camera: sapien.CameraEntity, name, dtype="float"):
    """Get texture from camera.

    Args:
        camera (sapien.CameraEntity): SAPIEN camera
        name (str): texture name
        dtype (str, optional): texture dtype, [float/uint32].
            Defaults to "float".

    Returns:
        np.ndarray: texture
    """
    if dtype == "float":
        return camera.get_float_texture(name)
    elif dtype == "uint32":
        return camera.get_uint32_texture(name)
    else:
        raise NotImplementedError(f"Unsupported texture type: {dtype}")


def get_texture_dlpack(camera: sapien.CameraEntity, name, dtype="float"):
    # dtype info is included already in the dl_tensor
    dl_tensor = camera.get_dl_tensor(name)
    return from_dlpack(dl_tensor).cpu().numpy()


if _USE_DLPACK:
    get_texture = get_texture_dlpack
else:
    get_texture = get_texture_naive


def get_camera_rgb(camera: sapien.CameraEntity, uint8=True):
    image = get_texture(camera, "Color")[..., :3]  # [H, W, 3]
    if uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def get_camera_depth(camera: sapien.CameraEntity):
    # position texture is in OpenGL frame, and thus depth is negative.
    # The unit is meter
    position = get_texture(camera, "Position")
    # NOTE(jigu): Depth is 0 if beyond camera far.
    depth = -position[..., [2]]  # [H, W, 1]
    # depth[position[..., [3]] == 1] = camera.far
    return depth


def get_camera_seg(camera: sapien.CameraEntity):
    seg = get_texture(camera, "Segmentation", "uint32")  # [H, W, 4]
    # channel 0 is visual id (mesh-level)
    # channel 1 is actor id (actor-level)
    return seg[..., :2]


def get_camera_images(
    camera: sapien.CameraEntity,
    rgb=True,
    depth=False,
    visual_seg=False,
    actor_seg=False,
) -> Dict[str, np.ndarray]:
    # Assume camera.take_picture() is called
    images = OrderedDict()
    if rgb:
        images["rgb"] = get_camera_rgb(camera)
    if depth:
        images["depth"] = get_camera_depth(camera)
    if visual_seg or actor_seg:
        seg = get_camera_seg(camera)
        if visual_seg:
            images["visual_seg"] = seg[..., 0:1]
        if actor_seg:
            images["actor_seg"] = seg[..., 1:2]
    return images


def get_camera_pcd(
    camera: sapien.CameraEntity,
    rgb=True,
    visual_seg=False,
    actor_seg=False,
) -> Dict[str, np.ndarray]:
    # Assume camera.take_picture() is called
    pcd = OrderedDict()
    # Each pixel is (x, y, z, z_buffer_depth) in OpenGL camera space
    position = get_texture(camera, "Position")  # [H, W, 4]
    position[..., 3] = position[..., 3] < 1
    pcd["xyzw"] = position.reshape(-1, 4)
    if rgb:
        pcd["rgb"] = get_camera_rgb(camera).reshape(-1, 3)
    if visual_seg or actor_seg:
        seg = get_camera_seg(camera)
        if visual_seg:
            pcd["visual_seg"] = seg[..., 0].reshape(-1, 1)
        if actor_seg:
            pcd["actor_seg"] = seg[..., 1].reshape(-1, 1)
    return pcd


@dataclass
class MountedCameraConfig:
    # Extrinsic parameters
    mount_link: str  # name of link to mount
    mount_p: List[float]  # position relative to link
    mount_q: List[float]  # quaternion relative to link
    hide_mount_link: bool  # whether to hide the mount link
    # Intrinsic parameters
    width: int
    height: int
    near: float
    far: float
    fx: float
    fy: float
    cx: float = None  # set to half width if None
    cy: float = None  # set to half height if None
    skew: float = 0.0

    def __post_init__(self):
        if self.cx is None:
            self.cx = self.width / 2
        if self.cy is None:
            self.cy = self.height / 2

    def build(self, articulation: sapien.Articulation, scene: sapien.Scene, name=""):
        camera_mount = get_entity_by_name(articulation.get_links(), self.mount_link)
        assert camera_mount is not None, self.mount_link
        mount_pose = sapien.Pose(self.mount_p, self.mount_q)
        camera = scene.add_mounted_camera(
            name,
            camera_mount,
            mount_pose,
            width=self.width,
            height=self.height,
            fovy=0,  # focal will be set later.
            near=self.near,
            far=self.far,
        )
        camera.set_perspective_parameters(
            near=self.near,
            far=self.far,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            skew=self.skew,
        )
        if self.hide_mount_link:
            set_actor_visibility(camera_mount, 0.0)
        return camera
