from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union

import numpy as np
import sapien
import sapien.render
import torch
from torch._tensor import Tensor

from mani_skill.render import SAPIEN_RENDER_SYSTEM, SHADER_CONFIGS, set_shader_pack
from mani_skill.utils.structs import Actor, Articulation, Link
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene

from mani_skill.utils import sapien_utils, visualization

from .base_sensor import BaseSensor, BaseSensorConfig

DEFAULT_TEXTURE_NAMES = ("Color", "PositionSegmentation")
if SAPIEN_RENDER_SYSTEM == "3.1":
    DEFAULT_TEXTURE_NAMES = ("Color", "PositionSegmentation")


@dataclass
class CameraConfig(BaseSensorConfig):

    uid: str
    """uid (str): unique id of the camera"""
    pose: Pose
    """Pose of the camera"""
    width: int
    """width (int): width of the camera"""
    height: int
    """height (int): height of the camera"""
    fov: float = None
    """The field of view of the camera. Either fov or intrinsic must be given"""
    near: float = 0.01
    """near (float): near plane of the camera"""
    far: float = 100
    """far (float): far plane of the camera"""
    intrinsic: Array = None
    """intrinsics matrix of the camera. Either fov or intrinsic must be given"""
    entity_uid: str = None
    """entity_uid (str, optional): unique id of the entity to mount the camera. Defaults to None."""
    mount: Union[Actor, Link] = None
    """the Actor or Link to mount the camera on top of. This means the global pose of the mounted camera is now mount.pose * local_pose"""
    texture_names: Optional[Sequence[str]] = None
    """texture_names (Sequence[str], optional): texture names to render."""
    shader_pack: str = "minimal"
    """The shader to use for rendering. Defaults to "minimal" which is the fastest rendering system with minimal GPU memory usage. There is also `default`."""

    def __post_init__(self):
        self.pose = Pose.create(self.pose)
        self.shader_config = SHADER_CONFIGS[self.shader_pack]

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(" + str(self.__dict__) + ")"


def update_camera_configs_from_dict(
    camera_configs: Dict[str, CameraConfig], config_dict: Dict[str, dict]
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
        self.texture_names = self.config.shader_config.texture_names

    def capture(self):
        self.camera.take_picture()

    def get_obs(self):
        images_dict = {}
        images = self.get_picture(self.texture_names)
        for img, name in zip(images, self.texture_names):
            images_dict[name] = img
        return images_dict

    def get_picture(self, name: str):
        if self.config.shader_pack == "minimal":
            rgb = (self.camera.get_picture("Color")[0][..., :3]).to(torch.uint8)
        else:
            rgb = (self.camera.get_picture("Color")[0][..., :3] * 255).to(torch.uint8)
        import matplotlib.pyplot as plt

        print(self.config.shader_pack, self.uid)
        # plt.imshow(rgb.cpu().numpy()[0]);plt.show()
        return self.camera.get_picture(name)

    def get_images(self) -> Tensor:
        return visualization.tile_images(
            visualization.observations_to_images(self.get_obs())
        )

    # TODO (stao): Computing camera parameters on GPU sim is not that fast, especially with mounted cameras and for model_matrix computation.
    def get_params(self):
        return dict(
            extrinsic_cv=self.camera.get_extrinsic_matrix(),
            cam2world_gl=self.camera.get_model_matrix(),
            intrinsic_cv=self.camera.get_intrinsic_matrix(),
        )
