import os
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Callable

import numpy as np
from sapien.core import Pose
from gym import spaces
import pyrealsense2 as rs
from real_robot.utils.realsense import RSDevice, get_connected_rs_devices


T_CV_CAM = np.array([[0, -1, 0, 0],
                     [0, 0, -1, 0],
                     [1, 0, 0, 0],
                     [0, 0, 0, 1]], dtype=np.float32)
T_CAM_CV = np.linalg.inv(T_CV_CAM)

CALIB_CAMERA_POSE_DIR = Path(
    os.getenv("HEC_CAMERA_POSE_DIR",
              "/rl_benchmark/real_robot/hec_camera_poses")
)
CALIB_CAMERA_POSES = {
    "front_camera": Pose().from_transformation_matrix(np.load(
        CALIB_CAMERA_POSE_DIR / "Tb_b2c_20230726_CSE4144_front.npy"
    ) @ T_CV_CAM)
}


class CameraConfig:
    def __init__(
        self,
        uid: str,
        device_sn: str,
        pose: Pose,
        width: int = 848,
        height: int = 480,
        preset="Default",
        depth_option_kwargs={},
        color_option_kwargs={},
        actor_pose_fn: Callable[..., Pose] = None,
    ):
        """Camera configuration.

        Args:
            uid (str): unique id of the camera
            device_sn (str): unique serial number of the camera
            pose (Pose): camera pose in world frame.
                         Format is is forward(x), left(y) and up(z)
                         If actor_pose_fn is not None,
                            this is pose relative to actor_pose
            width (int): width of the camera
            height (int): height of the camera
            preset (str): depth sensor presets
            depth_option_kwargs (dict): depth sensor optional keywords
            color_option_kwargs (dict): color sensor optional keywords
            actor_pose_fn (Callable, optional): function to get actor pose
                                                where the camera is mounted to.
                                                Defaults to None.
        """
        self.uid = uid
        self.device_sn = device_sn
        self.pose = pose
        self.width = width
        self.height = height

        self.preset = preset
        self.depth_option_kwargs = depth_option_kwargs
        self.color_option_kwargs = color_option_kwargs
        self.actor_pose_fn = actor_pose_fn

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(" + str(self.__dict__) + ")"


def parse_camera_cfgs(camera_cfgs):
    if isinstance(camera_cfgs, (tuple, list)):
        return OrderedDict([(cfg.uid, cfg) for cfg in camera_cfgs])
    elif isinstance(camera_cfgs, dict):
        return OrderedDict(camera_cfgs)
    elif isinstance(camera_cfgs, CameraConfig):
        return OrderedDict([(camera_cfgs.uid, camera_cfgs)])
    else:
        raise TypeError(type(camera_cfgs))


class Camera:
    """Wrapper for RealSense camera."""

    def __init__(self, camera_cfg: CameraConfig):
        self.camera_cfg = camera_cfg

        rs_config = self.get_rs_config(camera_cfg.width, camera_cfg.height)

        device = get_connected_rs_devices(camera_cfg.device_sn)
        self.device = RSDevice(
            device, rs_config, camera_cfg.preset,
            camera_cfg.depth_option_kwargs, camera_cfg.color_option_kwargs
        )
        self.device.start()

        self.frame_buffer = None  # for self.take_picture()

    def __del__(self):
        self.device.stop()

    @staticmethod
    def get_rs_config(width=848, height=480, fps=30) -> rs.config:
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height,
                             rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height,
                             rs.format.rgb8, fps)
        return config

    @property
    def uid(self):
        return self.camera_cfg.uid

    def take_picture(self):
        self.frame_buffer = self.device.wait_for_frames()

    def get_images(self, take_picture=False) -> Dict[str, np.ndarray]:
        """Get (raw) images from the camera.
        :return rgb: color image, [H, W, 3] np.uint8 array
        :return depth: depth image, [H, W, 1] np.float32 array
        """
        if take_picture:
            self.take_picture()

        rgb, depth = self.frame_buffer

        images = {
            "rgb": rgb,
            "depth": depth[..., None].astype(np.float32) / 1000.0
        }
        return images

    @property
    def pose(self) -> Pose:
        """Camera pose in world frame
        Format is is forward(x), left(y) and up(z)
        """
        if self.camera_cfg.actor_pose_fn is not None:
            return self.camera_cfg.actor_pose_fn() * self.camera_cfg.pose
        else:
            return self.camera_cfg.pose

    def get_extrinsic_matrix(self) -> np.ndarray:
        """Returns a 4x4 extrinsic camera matrix in OpenCV format
        right(x), down(y), forward(z)
        """
        return T_CV_CAM @ self.pose.inv().to_transformation_matrix()

    def get_model_matrix(self) -> np.ndarray:
        """Returns a 4x4 camera model matrix in OpenCV format
        right(x), down(y), forward(z)
        Note: this impl is different from sapien where the format is
              right(x), up(y), back(z)
        """
        return self.pose.to_transformation_matrix() @ T_CAM_CV

    def get_params(self):
        """Get camera parameters."""
        return dict(
            extrinsic_cv=self.get_extrinsic_matrix(),
            cam2world_cv=self.get_model_matrix(),
            intrinsic_cv=self.device.get_intrinsic_matrix(),
        )

    @property
    def observation_space(self) -> spaces.Dict:
        height, width = self.camera_cfg.height, self.camera_cfg.width
        obs_spaces = OrderedDict(
            rgb=spaces.Box(
                low=0, high=255, shape=(height, width, 3), dtype=np.uint8
            ),
            depth=spaces.Box(
                low=0, high=np.inf, shape=(height, width, 1), dtype=np.float32
            ),
        )
        return spaces.Dict(obs_spaces)
