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


@dataclass
class ObservationModeStruct:
    """A dataclass describing what observation data is being requested by the user"""

    state_dict: bool
    """whether to include state data which generally means including privileged information such as object poses"""
    state: bool
    """whether to include flattened state data which generally means including privileged information such as object poses"""
    visual: CameraObsTextures
    """textures to capture from cameras"""

    @property
    def use_state(self):
        """whether or not the environment should return ground truth/privileged information such as object poses"""
        return self.state or self.state_dict


ALL_VISUAL_TEXTURES = ["rgb", "depth", "segmentation", "position", "normal", "albedo"]
"""set of all standard textures that can come from cameras"""


def parse_obs_mode_to_struct(obs_mode: str) -> ObservationModeStruct:
    """Given user supplied observation mode, return a struct with the relevant textures that are to be captured"""
    # parse obs mode into a string of possible textures
    if obs_mode == "rgbd":
        return ObservationModeStruct(
            state_dict=False,
            state=False,
            visual=CameraObsTextures(
                rgb=True,
                depth=True,
                segmentation=False,
                position=False,
                normal=False,
                albedo=False,
            ),
        )
    elif obs_mode == "pointcloud":
        return ObservationModeStruct(
            state_dict=False,
            state=False,
            visual=CameraObsTextures(
                rgb=True,
                depth=False,
                segmentation=True,
                position=True,
                normal=False,
                albedo=False,
            ),
        )
    elif obs_mode == "sensor_data":
        return ObservationModeStruct(
            state_dict=False,
            state=False,
            visual=CameraObsTextures(
                rgb=True,
                depth=True,
                segmentation=True,
                position=True,
                normal=False,
                albedo=False,
            ),
        )
    else:
        # Parse obs mode into individual texture types
        textures = obs_mode.split("+")
        if "pointcloud" in textures:
            textures.remove("pointcloud")
            textures.append("position")
            textures.append("rgb")
            textures.append("segmentation")
        for texture in textures:
            if texture == "state" or texture == "state_dict" or texture == "none":
                # allows fetching privileged state data in addition to visual data.
                continue
            assert (
                texture in ALL_VISUAL_TEXTURES
            ), f"Invalid texture type '{texture}' requested in the obs mode '{obs_mode}'. Each individual texture must be one of {ALL_VISUAL_TEXTURES}"
        return ObservationModeStruct(
            state_dict="state_dict" in textures,
            state="state" in textures,
            visual=CameraObsTextures(
                rgb="rgb" in textures,
                depth="depth" in textures,
                segmentation="segmentation" in textures,
                position="position" in textures,
                normal="normal" in textures,
                albedo="albedo" in textures,
            ),
        )
