from dataclasses import dataclass, field
from typing import Any, Dict, List

import sapien

SAPIEN_RENDER_SYSTEM = "3.0"
try:
    # NOTE (stao): hacky way to determine which render system in sapien 3 is being used for testing purposes
    from sapien.wrapper.scene import get_camera_shader_pack

    SAPIEN_RENDER_SYSTEM = "3.1"
except:
    pass


@dataclass
class ShaderConfig:
    """simple shader config dataclass to determine which shader pack to use, textures to render, and any possible configurations for the shader pack"""

    shader_pack: str
    texture_names: List[str]
    shader_pack_config: Dict[str, Any] = field(default_factory=dict)


SHADER_CONFIGS = {
    "minimal": ShaderConfig(
        shader_pack="minimal", texture_names=["Color", "PositionSegmentation"]
    ),
    "default": ShaderConfig(
        shader_pack="default", texture_names=["Color", "PositionSegmentation"]
    ),
    "rt": ShaderConfig(
        shader_pack="rt",
        texture_names=["Color"],
        shader_pack_config={
            "ray_tracing_samples_per_pixel": 32,
            "ray_tracing_path_depth": 16,
            "ray_tracing_denoiser": "optix",
        },
    ),
    "rt-med": ShaderConfig(
        shader_pack="rt",
        texture_names=["Color"],
        shader_pack_config={
            "ray_tracing_samples_per_pixel": 4,
            "ray_tracing_path_depth": 3,
            "ray_tracing_denoiser": "optix",
        },
    ),
    "rt-fast": ShaderConfig(
        shader_pack="rt",
        texture_names=["Color"],
        shader_pack_config={
            "ray_tracing_samples_per_pixel": 2,
            "ray_tracing_path_depth": 1,
            "ray_tracing_denoiser": "optix",
        },
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
