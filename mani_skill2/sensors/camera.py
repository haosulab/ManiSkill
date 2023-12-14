from collections import OrderedDict
from typing import Dict, List, Sequence

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
from gymnasium import spaces

from mani_skill2.utils.sapien_utils import get_obj_by_name, hide_entity

from .base_sensor import BaseSensor, BaseSensorConfig


# TODO (stao): change this to a dataclass
class CameraConfig:
    def __init__(
        self,
        uid: str,
        p: List[float],
        q: List[float],
        width: int,
        height: int,
        fov: float,
        near: float,
        far: float,
        entity_uid: str = None,
        hide_link: bool = False,
        texture_names: Sequence[str] = ("Color", "Position"),
    ):
        """Camera configuration.

        Args:
            uid (str): unique id of the camera
            p (List[float]): position of the camera
            q (List[float]): quaternion of the camera
            width (int): width of the camera
            height (int): height of the camera
            fov (float): field of view of the camera
            near (float): near plane of the camera
            far (float): far plane of the camera
            entity_uid (str, optional): unique id of the entity to mount the camera. Defaults to None.
            hide_link (bool, optional): whether to hide the link to mount the camera. Defaults to False.
            texture_names (Sequence[str], optional): texture names to render. Defaults to ("Color", "Position").
        """
        self.uid = uid
        self.p = p
        self.q = q
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far

        self.actor_uid = entity_uid
        self.hide_link = hide_link
        self.texture_names = tuple(texture_names)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(" + str(self.__dict__) + ")"

    @property
    def pose(self):
        return sapien.Pose(self.p, self.q)

    @pose.setter
    def pose(self, pose: sapien.Pose):
        self.p = pose.p
        self.q = pose.q


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
    """Wrapper for sapien camera."""

    TEXTURE_DTYPE = {"Color": "float", "Position": "float", "Segmentation": "uint32"}

    def __init__(
        self,
        camera_cfg: CameraConfig,
        scene: sapien.Scene,
        renderer_type: str,
        articulation: physx.PhysxArticulation = None,
    ):
        super().__init__(sensor_type="camera")

        self.camera_cfg = camera_cfg
        self.renderer_type = renderer_type

        entity_uid = camera_cfg.actor_uid
        if entity_uid is None:
            self.entity = None
        else:
            if articulation is None:
                self.entity = get_obj_by_name(scene.get_entities(), entity_uid)
            else:
                self.entity = get_obj_by_name(
                    articulation.get_links(), entity_uid
                ).entity
            if self.entity is None:
                raise RuntimeError(f"Mount entity ({entity_uid}) is not found")

        # Add camera to scene. Add mounted one if a entity is given
        if self.entity is None:
            self.camera: sapien.render.RenderCameraComponent = scene.add_camera(
                camera_cfg.uid,
                camera_cfg.width,
                camera_cfg.height,
                camera_cfg.fov,
                camera_cfg.near,
                camera_cfg.far,
            )
            self.camera.local_pose = camera_cfg.pose
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
            hide_entity(self.entity)

        # Filter texture names according to renderer type if necessary (legacy for Kuafu)
        self.texture_names = camera_cfg.texture_names

    @property
    def uid(self):
        return self.camera_cfg.uid

    def take_picture(self):
        self.camera.take_picture()

    def get_images(self, take_picture=False):
        """Get (raw) images from the camera."""
        if take_picture:
            self.take_picture()

        if self.renderer_type == "client":
            return {}

        images = {}
        for name in self.texture_names:
            image = self.camera.get_picture(name)
            images[name] = image
        return images

    def get_params(self):
        """Get camera parameters."""
        return dict(
            extrinsic_cv=self.camera.get_extrinsic_matrix(),
            cam2world_gl=self.camera.get_model_matrix(),
            intrinsic_cv=self.camera.get_intrinsic_matrix(),
        )

    def get_obs(self):
        return self.get_images()

    @property
    def observation_space(self) -> spaces.Dict:
        obs_spaces = OrderedDict()
        height, width = self.camera.height, self.camera.width
        for name in self.texture_names:
            if name == "Color":
                obs_spaces[name] = spaces.Box(
                    low=0, high=1, shape=(height, width, 4), dtype=np.float32
                )
            elif name == "Position":
                obs_spaces[name] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(height, width, 4),
                    dtype=np.float32,
                )
            elif name == "Segmentation":
                obs_spaces[name] = spaces.Box(
                    low=np.iinfo(np.uint32).min,
                    high=np.iinfo(np.uint32).max,
                    shape=(height, width, 4),
                    dtype=np.uint32,
                )
            else:
                raise NotImplementedError(name)
        return spaces.Dict(obs_spaces)
