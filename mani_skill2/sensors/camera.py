from typing import Dict, List, Union, Sequence

import sapien.core as sapien
from mani_skill2.utils.sapien_utils import get_entity_by_name


class CameraConfig:
    def __init__(
        self,
        uuid: str,
        p: List[float],
        q: List[float],
        width: str,
        height: int,
        fov: float,
        near: float,
        far: float,
        articulation_uuid: str = None,
        actor_uuid: str = None,
        texture_names: Sequence[str] = ("Color",),
    ):
        self.uuid = uuid
        self.p = p
        self.q = q
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far

        self.articulation_uuid = articulation_uuid
        self.actor_uuid = actor_uuid
        self.texture_names = texture_names

    @property
    def pose(self):
        return sapien.Pose(self.p, self.q)

    def set_perspective_parameters(self):
        raise NotImplementedError


def update_camera_cfgs_from_dict(
    camera_cfgs: Dict[str, CameraConfig], cfg_dict: Dict[str, dict]
):
    # TODO(jigu): similar to urdf loader
    raise NotImplementedError


class Camera:
    TEXTURE_DTYPE = {"Color": "float"}

    def __init__(
        self, camera_cfg: CameraConfig, scene: sapien.Scene, renderer_type: str
    ):
        self.camera_cfg = camera_cfg
        self.renderer_type = renderer_type

        # TODO(jigu): more efficient way
        self.actor = self.get_mount_actor(
            scene, camera_cfg.articulation_uuid, camera_cfg.actor_uuid
        )

        # Add camera
        if self.actor is None:
            self.camera = scene.add_camera(
                camera_cfg.uuid,
                camera_cfg.width,
                camera_cfg.height,
                camera_cfg.fov,
                camera_cfg.near,
                camera_cfg.far,
            )
            self.camera.set_local_pose(camera_cfg.pose)
        else:
            self.camera = scene.add_mounted_camera(
                camera_cfg.uuid,
                self.actor,
                camera_cfg.pose,
                camera_cfg.width,
                camera_cfg.height,
                camera_cfg.fov,
                camera_cfg.near,
                camera_cfg.far,
            )

        # TODO(jigu): filter texture names according to renderer type and config
        self.texture_names = camera_cfg.texture_names

    @staticmethod
    def get_mount_actor(scene: sapien.Scene, articulation_uuid, actor_uuid):
        if actor_uuid is not None:
            if articulation_uuid is None:
                actor = get_entity_by_name(scene.get_all_actors(), actor_uuid)
            else:
                articulation = get_entity_by_name(
                    scene.get_all_articulations(), articulation_uuid
                )
                actor = get_entity_by_name(articulation.get_links(), actor_uuid)
        else:
            actor = None
        return actor

    def take_picture(self):
        self.camera.take_picture()

    def get_images(self, take_picture=False):
        if take_picture:
            self.take_picture()
        images = {}
        for name in self.texture_names:
            dtype = self.TEXTURE_DTYPE[name]
            if dtype == "float":
                image = self.camera.get_float_texture(name)
            elif dtype == "uint32":
                image = self.camera.get_uint32_texture(name)
            else:
                raise NotImplementedError(dtype)
            images[name] = image
        return images
