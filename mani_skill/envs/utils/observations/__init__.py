from dataclasses import dataclass

from .observations import *


@dataclass
class CameraObsTextures:
    rgb: bool
    depth: bool
    segmentation: bool


def parse_visual_obs_mode_to_struct(obs_mode: str) -> CameraObsTextures:
    """Given user supplied observation mode, return a struct with the relevant textures that are to be captured"""
    if obs_mode == "rgb":
        return CameraObsTextures(rgb=True, depth=False, segmentation=False)
    elif obs_mode == "rgbd":
        return CameraObsTextures(rgb=True, depth=True, segmentation=False)
    elif obs_mode == "rgb+depth":
        return CameraObsTextures(rgb=True, depth=True, segmentation=False)
    elif obs_mode == "rgb+depth+segmentation":
        return CameraObsTextures(rgb=True, depth=True, segmentation=True)
    elif obs_mode == "rgb+segmentation":
        return CameraObsTextures(rgb=True, depth=False, segmentation=True)
    elif obs_mode == "depth+segmentation":
        return CameraObsTextures(rgb=False, depth=True, segmentation=True)
    else:
        return None
