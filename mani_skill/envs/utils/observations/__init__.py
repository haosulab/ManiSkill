from dataclasses import dataclass

from .observations import *


@dataclass
class CameraObsTextures:
    rgb: bool
    depth: bool
    segmentation: bool
    position: bool
    normal: bool
    albedo: bool


ALL_TEXTURES = ["rgb", "depth", "segmentation", "position", "normal", "albedo"]
"""set of all standard textures that can come from cameras"""


def parse_visual_obs_mode_to_struct(obs_mode: str) -> CameraObsTextures:
    """Given user supplied observation mode, return a struct with the relevant textures that are to be captured"""
    # parse obs mode into a string of possible textures
    if obs_mode == "rgbd":
        return CameraObsTextures(
            rgb=True,
            depth=True,
            segmentation=False,
            position=False,
            normal=False,
            albedo=False,
        )
    elif obs_mode == "pointcloud":
        return CameraObsTextures(
            rgb=True,
            depth=False,
            segmentation=True,
            position=True,
            normal=False,
            albedo=False,
        )
    elif obs_mode == "sensor_data":
        return CameraObsTextures(
            rgb=True,
            depth=True,
            segmentation=True,
            position=True,
            normal=False,
            albedo=False,
        )
    elif obs_mode in ["state", "state_dict", "none"]:
        return None
    else:
        # Parse obs mode into individual texture types
        textures = obs_mode.split("+")
        for texture in textures:
            assert (
                texture in ALL_TEXTURES
            ), f"Invalid texture type '{texture}' requested in the obs mode '{obs_mode}'. Each individual texture must be one of {ALL_TEXTURES}"
        return CameraObsTextures(
            rgb="rgb" in textures,
            depth="depth" in textures,
            segmentation="segmentation" in textures,
            position="position" in textures,
            normal="normal" in textures,
            albedo="albedo" in textures,
        )
