from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import sapien
import torch

from mani_skill.render.version import SAPIEN_RENDER_SYSTEM


@dataclass
class ShaderConfig:
    """simple shader config dataclass to determine which shader pack to use, textures to render, and any possible configurations for the shader pack. Can be used as part of the CameraConfig
    to further customize the camera output.

    A shader config must define which shader pack to use, and which textures to consider rendering. Additional shader pack configs can be passed which are specific to the shader config itself
    and can modify shader settings.

    Texture transforms must be defined and are used to process the texture data into more standard formats for use. Some textures might be combined textures (e.g. depth+segmentation together)
    due to shader optimizations. texture transforms must then split these combined textures back into their component parts.

    The standard image modalities and expected dtypes/shapes are:
        - rgb (torch.uint8, shape: [H, W, 3])
        - depth (torch.int16, shape: [H, W])
        - segmentation (torch.int16, shape: [H, W])
        - position (torch.float32, shape: [H, W, 3]) (infinite points have segmentation == 0)
    """

    shader_pack: str
    texture_names: Dict[str, List[str]] = field(default_factory=dict)
    """dictionary mapping shader texture names to the image modalities that are rendered. e.g. Color, Depth, Segmentation, etc."""
    shader_pack_config: Dict[str, Any] = field(default_factory=dict)
    """configs for the shader pack. for e.g. the ray tracing shader you can configure the denoiser, samples per pixel, etc."""

    texture_transforms: Dict[
        str, Callable[[torch.Tensor], Dict[str, torch.Tensor]]
    ] = field(default_factory=dict)
    """texture transform functions that map each texture name to a function that converts the texture data into one or more standard image modalities. The return type should be a
    dictionary with keys equal to the names of standard image modalities and values equal to the transformed data"""


def default_position_texture_transform(data: torch.Tensor):
    position = (data[..., :3] * 1000).to(torch.int16)
    depth = -position[..., [2]]
    return {
        "depth": depth,
        "position": position,
    }


rt_texture_transforms = {
    "Color": lambda data: {"rgb": (data[..., :3] * 255).to(torch.uint8)},
    "Position": default_position_texture_transform,
    # note in default shader pack, 0 is visual shape / mesh, 1 is actor/link level, 2 is parallel scene ID, 3 is unused
    "Segmentation": lambda data: {"segmentation": data[..., 1][..., None]},
    "Normal": lambda data: {"normal": data[..., :3]},
    "Albedo": lambda data: {"albedo": (data[..., :3] * 255).to(torch.uint8)},
}
rt_texture_names = {
    "Color": ["rgb"],
    "Position": ["position", "depth"],
    "Segmentation": ["segmentation"],
    "Normal": ["normal"],
    "Albedo": ["albedo"],
}


PREBUILT_SHADER_CONFIGS = {
    "minimal": ShaderConfig(
        shader_pack="minimal",
        texture_names={
            "Color": ["rgb"],
            "PositionSegmentation": ["position", "depth", "segmentation"],
        },
        texture_transforms={
            "Color": lambda data: {"rgb": data[..., :3]},
            "PositionSegmentation": lambda data: {
                "position": data[
                    ..., :3
                ],  # position for minimal is in millimeters and is uint16
                "depth": -data[..., [2]],
                "segmentation": data[..., [3]],
            },
        },
    ),
    "default": ShaderConfig(
        shader_pack="default",
        texture_names={
            "Color": ["rgb"],
            "Position": ["position", "depth"],
            "Segmentation": ["segmentation"],
            "Normal": ["normal"],
            "Albedo": ["albedo"],
        },
        texture_transforms={
            "Color": lambda data: {"rgb": (data[..., :3] * 255).to(torch.uint8)},
            "Position": default_position_texture_transform,
            # note in default shader pack, 0 is visual shape / mesh, 1 is actor/link level, 2 is parallel scene ID, 3 is unused
            "Segmentation": lambda data: {"segmentation": data[..., 1][..., None]},
            "Normal": lambda data: {"normal": data[..., :3]},
            "Albedo": lambda data: {"albedo": (data[..., :3] * 255).to(torch.uint8)},
        },
    ),
    "rt": ShaderConfig(
        shader_pack="rt",
        texture_names=rt_texture_names,
        shader_pack_config={
            "ray_tracing_samples_per_pixel": 32,
            "ray_tracing_path_depth": 16,
            "ray_tracing_denoiser": "optix",
        },
        texture_transforms=rt_texture_transforms,
    ),
    "rt-med": ShaderConfig(
        shader_pack="rt",
        texture_names=rt_texture_names,
        shader_pack_config={
            "ray_tracing_samples_per_pixel": 4,
            "ray_tracing_path_depth": 3,
            "ray_tracing_denoiser": "optix",
        },
        texture_transforms=rt_texture_transforms,
    ),
    "rt-fast": ShaderConfig(
        shader_pack="rt",
        texture_names=rt_texture_names,
        shader_pack_config={
            "ray_tracing_samples_per_pixel": 2,
            "ray_tracing_path_depth": 1,
            "ray_tracing_denoiser": "optix",
        },
        texture_transforms=rt_texture_transforms,
    ),
}
"""pre-defined shader configs"""


def set_shader_pack(shader_config: ShaderConfig):
    """sets a global shader pack for cameras. Used only for the 3.0 SAPIEN rendering system"""
    if SAPIEN_RENDER_SYSTEM == "3.0":
        sapien.render.set_camera_shader_dir(shader_config.shader_pack)
        if shader_config.shader_pack == "minimal":
            sapien.render.set_camera_shader_dir("minimal")
            sapien.render.set_picture_format("Color", "r8g8b8a8unorm")
            sapien.render.set_picture_format("ColorRaw", "r8g8b8a8unorm")
            sapien.render.set_picture_format("PositionSegmentation", "r16g16b16a16sint")
        if shader_config.shader_pack == "default":
            sapien.render.set_camera_shader_dir("default")
            sapien.render.set_picture_format("Color", "r32g32b32a32sfloat")
            sapien.render.set_picture_format("ColorRaw", "r32g32b32a32sfloat")
            sapien.render.set_picture_format(
                "PositionSegmentation", "r32g32b32a32sfloat"
            )
        if shader_config.shader_pack[:2] == "rt":
            sapien.render.set_ray_tracing_samples_per_pixel(
                shader_config.shader_pack_config["ray_tracing_samples_per_pixel"]
            )
            sapien.render.set_ray_tracing_path_depth(
                shader_config.shader_pack_config["ray_tracing_path_depth"]
            )
            sapien.render.set_ray_tracing_denoiser(
                shader_config.shader_pack_config["ray_tracing_denoiser"]
            )
    elif SAPIEN_RENDER_SYSTEM == "3.1":
        # sapien.render.set_camera_shader_pack_name would set a global default
        pass
