from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import sapien
import sapien.render
import torch
from torch._tensor import Tensor

from mani_skill.render import (
    PREBUILT_SHADER_CONFIGS,
    SAPIEN_RENDER_SYSTEM,
    ShaderConfig,
    set_shader_pack,
)
from mani_skill.utils.structs import Actor, Articulation, Link
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene

from mani_skill.utils import sapien_utils, visualization

from .base_sensor import BaseSensor, BaseSensorConfig


@dataclass
class CameraConfig(BaseSensorConfig):

    uid: str
    """uid (str): unique id of the camera"""
    pose: Pose
    """Pose of the camera"""
    width: int
    """width of the camera"""
    height: int
    """height of the camera"""
    fov: float = None
    """The field of view of the camera. Either fov or intrinsic must be given"""
    near: float = 0.01
    """near plane of the camera"""
    far: float = 100
    """far plane of the camera"""
    intrinsic: Array = None
    """intrinsics matrix of the camera. Either fov or intrinsic must be given"""
    entity_uid: Optional[str] = None
    """unique id of the entity to mount the camera. Defaults to None. Only used by agent classes that want to define mounted cameras."""
    mount: Union[Actor, Link] = None
    """the Actor or Link to mount the camera on top of. This means the global pose of the mounted camera is now mount.pose * local_pose"""
    shader_pack: Optional[str] = "minimal"
    """The shader to use for rendering. Defaults to "minimal" which is the fastest rendering system with minimal GPU memory usage. There is also ``default`` and ``rt``."""
    shader_config: Optional[ShaderConfig] = None
    """The shader config to use for rendering. If None, the shader_pack will be used to search amongst prebuilt shader configs to create a ShaderConfig."""

    def __post_init__(self):
        self.pose = Pose.create(self.pose)
        if self.shader_config is None:
            self.shader_config = PREBUILT_SHADER_CONFIGS[self.shader_pack]
        else:
            self.shader_pack = self.shader_config.shader_pack

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(" + str(self.__dict__) + ")"


def update_camera_configs_from_dict(
    camera_configs: dict[str, CameraConfig], config_dict: dict[str, dict]
):
    # Update CameraConfig to StereoDepthCameraConfig
    if config_dict.pop("use_stereo_depth", False):
        from .depth_camera import StereoDepthCameraConfig  # fmt: skip
        for name, config in camera_configs.items():
            camera_configs[name] = StereoDepthCameraConfig.fromCameraConfig(config)

    # First, apply global configuration
    for k, v in config_dict.items():
        if k in camera_configs:
            continue
        for config in camera_configs.values():
            if not hasattr(config, k):
                raise AttributeError(f"{k} is not a valid attribute of CameraConfig")
            else:
                if k == "shader_pack":
                    config.shader_config = None
                setattr(config, k, v)
    # Then, apply camera-specific configuration
    for name, v in config_dict.items():
        if name not in camera_configs:
            continue

        # Update CameraConfig to StereoDepthCameraConfig
        if v.pop("use_stereo_depth", False):
            from .depth_camera import StereoDepthCameraConfig  # fmt: skip
            config = camera_configs[name]
            camera_configs[name] = StereoDepthCameraConfig.fromCameraConfig(config)

        config = camera_configs[name]
        for kk in v:
            if kk == "shader_pack":
                config.shader_config = None
            assert hasattr(config, kk), f"{kk} is not a valid attribute of CameraConfig"
        v = copy.deepcopy(v)
        # for json serailizable gym.make args, user has to pass a list, not a Pose object.
        if "pose" in v and isinstance(v["pose"], list):
            v["pose"] = sapien.Pose(v["pose"][:3], v["pose"][3:])
        config.__dict__.update(v)
    for config in camera_configs.values():
        config.__post_init__()


def parse_camera_configs(camera_configs):
    if isinstance(camera_configs, (tuple, list)):
        return dict([(config.uid, config) for config in camera_configs])
    elif isinstance(camera_configs, dict):
        return dict(camera_configs)
    elif isinstance(camera_configs, CameraConfig):
        return dict([(camera_configs.uid, camera_configs)])
    else:
        raise TypeError(type(camera_configs))


class Camera(BaseSensor):
    """Implementation of the Camera sensor which uses the sapien Camera."""

    config: CameraConfig

    def __init__(
        self,
        camera_config: CameraConfig,
        scene: ManiSkillScene,
        articulation: Articulation = None,
    ):
        super().__init__(config=camera_config)
        entity_uid = camera_config.entity_uid
        if camera_config.mount is not None:
            self.entity = camera_config.mount
        elif entity_uid is None:
            self.entity = None
        else:
            if articulation is None:
                pass
            else:
                # if given an articulation and entity_uid (as a string), find the correct link to mount on
                # this is just for convenience so robot configurations can pick link to mount to by string/id
                self.entity = sapien_utils.get_obj_by_name(
                    articulation.get_links(), entity_uid
                )
            if self.entity is None:
                raise RuntimeError(f"Mount entity ({entity_uid}) is not found")

        intrinsic = camera_config.intrinsic
        assert (camera_config.fov is None and intrinsic is not None) or (
            camera_config.fov is not None and intrinsic is None
        )

        # Add camera to scene. Add mounted one if a entity is given
        set_shader_pack(self.config.shader_config)
        if self.entity is None:
            self.camera = scene.add_camera(
                name=camera_config.uid,
                pose=camera_config.pose,
                width=camera_config.width,
                height=camera_config.height,
                fovy=camera_config.fov,
                intrinsic=intrinsic,
                near=camera_config.near,
                far=camera_config.far,
            )
        else:
            self.camera = scene.add_camera(
                name=camera_config.uid,
                mount=self.entity,
                pose=camera_config.pose,
                width=camera_config.width,
                height=camera_config.height,
                fovy=camera_config.fov,
                intrinsic=intrinsic,
                near=camera_config.near,
                far=camera_config.far,
            )
        # Filter texture names according to renderer type if necessary (legacy for Kuafu)

    def capture(self):
        self.camera.take_picture()

    def get_obs(
        self,
        rgb: bool = True,
        depth: bool = True,
        position: bool = True,
        segmentation: bool = True,
        normal: bool = False,
        albedo: bool = False,
        apply_texture_transforms: bool = True,
    ):
        images_dict = {}
        # determine which textures are needed to get the desired modalities
        required_texture_names = []
        for (
            texture_name,
            output_modalities,
        ) in self.config.shader_config.texture_names.items():
            if rgb and "rgb" in output_modalities:
                required_texture_names.append(texture_name)
            if depth and "depth" in output_modalities:
                required_texture_names.append(texture_name)
            if position and "position" in output_modalities:
                required_texture_names.append(texture_name)
            if segmentation and "segmentation" in output_modalities:
                required_texture_names.append(texture_name)
            if normal and "normal" in output_modalities:
                required_texture_names.append(texture_name)
            if albedo and "albedo" in output_modalities:
                required_texture_names.append(texture_name)
        required_texture_names = list(set(required_texture_names))

        # fetch the image data
        output_textures = self.camera.get_picture(required_texture_names)
        for texture_name, texture in zip(required_texture_names, output_textures):
            if apply_texture_transforms:
                images_dict |= self.config.shader_config.texture_transforms[
                    texture_name
                ](texture)
            else:
                images_dict[texture_name] = texture
        if not rgb and "rgb" in images_dict:
            del images_dict["rgb"]
        if not depth and "depth" in images_dict:
            del images_dict["depth"]
        if not position and "position" in images_dict:
            del images_dict["position"]
        if not segmentation and "segmentation" in images_dict:
            del images_dict["segmentation"]
        if not normal and "normal" in images_dict:
            del images_dict["normal"]
        if not albedo and "albedo" in images_dict:
            del images_dict["albedo"]
        return images_dict

    def get_images(self, obs) -> Tensor:
        return camera_observations_to_images(obs)

    # TODO (stao): Computing camera parameters on GPU sim is not that fast, especially with mounted cameras and for model_matrix computation.
    def get_params(self):
        return dict(
            extrinsic_cv=self.camera.get_extrinsic_matrix(),
            cam2world_gl=self.camera.get_model_matrix(),
            intrinsic_cv=self.camera.get_intrinsic_matrix(),
        )


def normalize_depth(depth, min_depth=0, max_depth=None):
    if min_depth is None:
        min_depth = depth.min()
    if max_depth is None:
        max_depth = depth.max()
    depth = (depth - min_depth) / (max_depth - min_depth)
    depth = depth.clip(0, 1)
    return depth


def camera_observations_to_images(
    observations: dict[str, torch.Tensor], max_depth=None
) -> list[Array]:
    """Parse images from camera observations."""
    images = dict()
    for key in observations:
        if "rgb" in key or "Color" in key:
            rgb = observations[key][..., :3]
            if torch is not None and rgb.dtype == torch.float:
                rgb = torch.clip(rgb * 255, 0, 255).to(torch.uint8)
            images[key] = rgb
        elif "depth" in key or "position" in key:
            depth = observations[key]
            if "position" in key:  # [H, W, 4]
                depth = -depth[..., 2:3]
            # [H, W, 1]
            depth = normalize_depth(depth, max_depth=max_depth)
            depth = (depth * 255).clip(0, 255)

            depth = depth.to(torch.uint8)
            depth = torch.repeat_interleave(depth, 3, dim=-1)
            images[key] = depth
        elif "segmentation" in key:
            seg = observations[key]  # [H, W, 1]
            assert seg.ndim == 4 and seg.shape[-1] == 1, seg.shape
            # A heuristic way to colorize labels
            seg = (seg * torch.tensor([11, 61, 127], device=seg.device)).to(torch.uint8)
            images[key] = seg
    return images
