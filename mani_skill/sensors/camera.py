from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Sequence, Union

import sapien
import sapien.render

from mani_skill.utils.structs import Actor, Articulation, Link
from mani_skill.utils.structs.pose import Pose

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene

from mani_skill.utils import sapien_utils

from .base_sensor import BaseSensor, BaseSensorConfig


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
    fov: float
    """fov (float): field of view of the camera"""
    near: float
    """near (float): near plane of the camera"""
    far: float
    """far (float): far plane of the camera"""
    entity_uid: str = None
    """entity_uid (str, optional): unique id of the entity to mount the camera. Defaults to None."""
    mount: Union[Actor, Link] = None
    """the Actor or Link to mount the camera on top of. This means the global pose of the mounted camera is now mount.pose * local_pose"""
    hide_link: bool = False
    """hide_link (bool, optional): whether to hide the link to mount the camera. Defaults to False."""
    texture_names: Sequence[str] = ("Color", "PositionSegmentation")
    """texture_names (Sequence[str], optional): texture names to render. Defaults to ("Color", "PositionSegmentation"). Note that the renderign speed will not really change if you remove PositionSegmentation"""

    def __post_init__(self):
        self.pose = Pose.create(self.pose)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(" + str(self.__dict__) + ")"


def update_camera_cfgs_from_dict(
    camera_cfgs: Dict[str, CameraConfig], cfg_dict: Dict[str, dict]
):
    # Update CameraConfig to StereoDepthCameraConfig
    if cfg_dict.pop("use_stereo_depth", False):
        from .depth_camera import StereoDepthCameraConfig  # fmt: skip
        for name, cfg in camera_cfgs.items():
            camera_cfgs[name] = StereoDepthCameraConfig.fromCameraConfig(cfg)

    # First, apply global configuration
    for k, v in cfg_dict.items():
        if k in camera_cfgs:
            continue
        for cfg in camera_cfgs.values():
            if k == "add_segmentation":
                # TODO (stao): doesn't work this way anymore
                cfg.texture_names += ("Segmentation",)
            elif not hasattr(cfg, k):
                raise AttributeError(f"{k} is not a valid attribute of CameraConfig")
            else:
                setattr(cfg, k, v)
    # Then, apply camera-specific configuration
    for name, v in cfg_dict.items():
        if name not in camera_cfgs:
            continue

        # Update CameraConfig to StereoDepthCameraConfig
        if v.pop("use_stereo_depth", False):
            from .depth_camera import StereoDepthCameraConfig  # fmt: skip
            cfg = camera_cfgs[name]
            camera_cfgs[name] = StereoDepthCameraConfig.fromCameraConfig(cfg)

        cfg = camera_cfgs[name]
        for kk in v:
            assert hasattr(cfg, kk), f"{kk} is not a valid attribute of CameraConfig"
        cfg.__dict__.update(v)


def parse_camera_cfgs(camera_cfgs):
    if isinstance(camera_cfgs, (tuple, list)):
        return OrderedDict([(cfg.uid, cfg) for cfg in camera_cfgs])
    elif isinstance(camera_cfgs, dict):
        return OrderedDict(camera_cfgs)
    elif isinstance(camera_cfgs, CameraConfig):
        return OrderedDict([(camera_cfgs.uid, camera_cfgs)])
    else:
        raise TypeError(type(camera_cfgs))


class Camera(BaseSensor):
    """Implementation of the Camera sensor which uses the sapien Camera."""

    def __init__(
        self,
        camera_cfg: CameraConfig,
        scene: ManiSkillScene,
        articulation: Articulation = None,
    ):
        super().__init__(cfg=camera_cfg)

        self.camera_cfg = camera_cfg

        entity_uid = camera_cfg.entity_uid
        if camera_cfg.mount is not None:
            self.entity = camera_cfg.mount
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

        # Add camera to scene. Add mounted one if a entity is given
        if self.entity is None:
            self.camera = scene.add_camera(
                camera_cfg.uid,
                camera_cfg.pose,
                camera_cfg.width,
                camera_cfg.height,
                camera_cfg.fov,
                camera_cfg.near,
                camera_cfg.far,
            )
        else:
            self.camera = scene.add_mounted_camera(
                camera_cfg.uid,
                self.entity,
                camera_cfg.pose,
                camera_cfg.width,
                camera_cfg.height,
                camera_cfg.fov,
                camera_cfg.near,
                camera_cfg.far,
            )

        if camera_cfg.hide_link:
            # TODO (stao): re-implement this
            from mani_skill import logger

            logger.warn(
                "camera hide_link option is not implemented yet so this won't be hidden"
            )

        # Filter texture names according to renderer type if necessary (legacy for Kuafu)
        self.texture_names = camera_cfg.texture_names

    def capture(self):
        self.camera.take_picture()

    def get_obs(self):
        images = {}
        for name in self.texture_names:
            image = self.get_picture(name)
            images[name] = image
        return images

    def get_picture(self, name: str):
        return self.camera.get_picture(name)

    # TODO (stao): Computing camera parameters on GPU sim is not that fast, especially with mounted cameras and for model_matrix computation.
    def get_params(self):
        return dict(
            extrinsic_cv=self.camera.get_extrinsic_matrix(),
            cam2world_gl=self.camera.get_model_matrix(),
            intrinsic_cv=self.camera.get_intrinsic_matrix(),
        )
