from collections import OrderedDict
from typing import Dict, List, Sequence

import numpy as np
import sapien.core as sapien
from gym import spaces
from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig

from mani_skill2.utils.sapien_utils import get_entity_by_name

from .camera import Camera, CameraConfig


class DepthCameraConfig(CameraConfig):
    def __init__(
        self,
        uid: str,
        p: List[float],
        q: List[float],
        actor_uid: str = None,
        hide_link: bool = False,
        texture_names: Sequence[str] = ("Color", "Position"),
    ):
        self.uid = uid
        self.p = p
        self.q = q

        self.actor_uid = actor_uid
        self.hide_link = hide_link
        self.texture_names = tuple(texture_names)

    @classmethod
    def fromCameraConfig(cls, cfg: CameraConfig):
        return cls(
            cfg.uid,
            cfg.p,
            cfg.q,
            actor_uid=cfg.actor_uid,
            hide_link=cfg.hide_link,
            texture_names=cfg.texture_names,
        )


class DepthCamera(Camera):
    def __init__(
        self,
        camera_cfg: DepthCameraConfig,
        scene: sapien.Scene,
        renderer_type: str,
        articulation: sapien.Articulation = None,
    ):
        self.camera_cfg = camera_cfg
        assert renderer_type == "sapien", renderer_type
        self.renderer_type = renderer_type

        actor_uid = camera_cfg.actor_uid
        if actor_uid is None:
            self.actor = None
        else:
            if articulation is None:
                self.actor = get_entity_by_name(scene.get_all_actors(), actor_uid)
            else:
                self.actor = get_entity_by_name(articulation.get_links(), actor_uid)
            if self.actor is None:
                raise RuntimeError(f"Mount actor ({actor_uid}) is not found")

        # Add camera
        sensor_config = StereoDepthSensorConfig()
        if self.actor is None:
            self.camera = StereoDepthSensor(
                camera_cfg.uid, scene, sensor_config, mount=self.actor
            )
            self.camera.set_pose(camera_cfg.pose)
        else:
            self.camera = StereoDepthSensor(
                camera_cfg.uid,
                scene,
                sensor_config,
                mount=self.actor,
                pose=camera_cfg.pose,
            )

        if camera_cfg.hide_link:
            self.actor.hide_visual()

        # Filter texture names according to renderer type if necessary (legacy for Kuafu)
        self.texture_names = camera_cfg.texture_names

    def get_images(self, take_picture=False):
        """Get (raw) images from the camera."""
        if take_picture:
            self.take_picture()

        if self.renderer_type == "client":
            return {}

        images = {}
        for name in self.texture_names:
            if name == "Color":
                image = self.camera._cam_rgb.get_float_texture("Color")
            elif name == "depth":
                self.camera.compute_depth()
                image = self.camera.get_depth()[..., None]
            elif name == "Position":
                self.camera.compute_depth()
                position = self.camera._cam_rgb.get_float_texture("Position")
                depth = self.camera.get_depth()
                position[..., 2] = -depth
                image = position
            elif name == "Segmentation":
                image = self.camera._cam_rgb.get_uint32_texture("Segmentation")
            else:
                raise NotImplementedError(name)
            images[name] = image
        return images

    def get_params(self):
        """Get camera parameters."""
        return dict(
            extrinsic_cv=self.camera._cam_rgb.get_extrinsic_matrix(),
            cam2world_gl=self.camera._cam_rgb.get_model_matrix(),
            intrinsic_cv=self.camera._cam_rgb.get_intrinsic_matrix(),
        )

    @property
    def observation_space(self) -> spaces.Dict:
        obs_spaces = OrderedDict()
        width, height = self.camera._cam_rgb.width, self.camera._cam_rgb.height
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
