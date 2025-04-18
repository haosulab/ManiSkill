import numpy as np
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

REALSENSE_DEPTH_FOV_VERTICAL_RAD = 58.0 * np.pi / 180
REALSENSE_DEPTH_FOV_HORIZONTAL_RAD = 87.0 * np.pi / 180

SHADER = "default"

def get_camera_configs(xy_offset, z_offset, target: tuple[float, float, float], camera_width, camera_height):
    pose_center = sapien_utils.look_at(eye=[xy_offset, 0,  z_offset], target=target)
    pose_left = sapien_utils.look_at(eye=[0.0, -xy_offset, z_offset], target=target)
    pose_right = sapien_utils.look_at(eye=[0.0, xy_offset, z_offset], target=target)
    return [
        CameraConfig(
            uid="camera_center",
            pose=pose_center,
            width=camera_width,
            height=camera_height,
            fov=REALSENSE_DEPTH_FOV_VERTICAL_RAD,
            near=0.01,
            far=100,
            shader_pack=SHADER,
        ),
        CameraConfig(
            uid="camera_left",
            pose=pose_left,
            width=camera_width,
            height=camera_height,
            fov=REALSENSE_DEPTH_FOV_VERTICAL_RAD,
            near=0.01,
            far=100,
            shader_pack=SHADER,
        ),
        CameraConfig(
            uid="camera_right",
            pose=pose_right,
            width=camera_width,
            height=camera_height,
            fov=REALSENSE_DEPTH_FOV_VERTICAL_RAD,
            near=0.01,
            far=100,
            shader_pack=SHADER,
        )]
