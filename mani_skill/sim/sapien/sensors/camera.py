from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mani_skill.render import PREBUILT_SHADER_CONFIGS, ShaderConfig, set_shader_pack
from mani_skill.sim.sapien.structs.actor import SapienActor
from mani_skill.sim.sapien.structs.articulation import SapienArticulation
from mani_skill.sim.sapien.structs.link import SapienLink
from mani_skill.sim.sensors.camera import Camera, CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import Pose

if TYPE_CHECKING:
    from mani_skill.sim.sapien.sim import SapienSim


@dataclass
class SapienCameraConfig(CameraConfig):
    shader_pack: str = "minimal"
    """The shader to use for rendering. Defaults to "minimal" which is the fastest rendering system with minimal GPU memory usage. There is also ``default`` and ``rt``."""
    shader_config: ShaderConfig | None = None
    """The shader config to use for rendering. If None, the shader_pack will be used to search amongst prebuilt shader configs to create a ShaderConfig."""
    mount: SapienActor | SapienLink | None = None

    def __post_init__(self):
        self.pose = Pose.create(self.pose)
        if self.shader_config is None:
            self.shader_config = PREBUILT_SHADER_CONFIGS[self.shader_pack]
        else:
            self.shader_pack = self.shader_config.shader_pack


class SapienCamera(Camera):
    def __init__(
        self,
        camera_config: SapienCameraConfig,
        sim: SapienSim,
        articulation: SapienArticulation | None = None,
    ):
        super().__init__(camera_config, sim, articulation)
        self._shader_config = cast(ShaderConfig, camera_config.shader_config)
        entity_uid = camera_config.entity_uid
        if camera_config.mount is not None:
            self.entity = camera_config.mount
        elif entity_uid is None:
            self.entity = cast(SapienActor | SapienLink, None)
        else:
            if articulation is None:
                pass
            else:
                # if given an articulation and entity_uid (as a string), find the correct link to mount on
                # this is just for convenience so robot configurations can pick link to mount to by string/id
                self.entity = cast(
                    SapienActor | SapienLink,
                    sapien_utils.get_obj_by_name(articulation.get_links(), entity_uid),
                )
            if self.entity is None:
                raise RuntimeError(f"Mount entity ({entity_uid}) is not found")

        intrinsic = camera_config.intrinsic
        assert (camera_config.fov is None and intrinsic is not None) or (
            camera_config.fov is not None and intrinsic is None
        )

        # Add camera to scene. Add mounted one if a entity is given
        set_shader_pack(self._shader_config)
        if self.entity is None:
            self.camera = sim.add_camera(
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
            self.camera = sim.add_camera(
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
        ) in self._shader_config.texture_names.items():
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
                images_dict |= self._shader_config.texture_transforms[texture_name](
                    texture
                )
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

    def get_images(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Get RGB versions of images in the observation dictionary
        """
        return camera_observations_to_images(obs)

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
) -> dict[str, torch.Tensor]:
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
