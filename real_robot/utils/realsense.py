from typing import List, Tuple, Union, Optional

import pyrealsense2 as rs
import numpy as np

from .logger import get_logger


logger = get_logger("realsense.py")
rs_devices = None  # {device_sn: rs.device}


def get_connected_rs_devices(
    device_sn: Union[str, List[str]] = None
) -> Union[rs.device, List[rs.device]]:
    """Returns list of connected RealSense devices
    :param device_sn: list of serial numbers of devices to get. If not None,
                      only return those devices in matching order.
    :return devices: list of rs.device
    """
    global rs_devices

    if rs_devices is None:
        rs_devices = {}
        for d in rs.context().devices:
            name = d.get_info(rs.camera_info.name)
            if name.lower() != 'platform camera':
                serial = d.get_info(rs.camera_info.serial_number)
                fw_version = d.get_info(rs.camera_info.firmware_version)
                usb_type = d.get_info(rs.camera_info.usb_type_descriptor)

                logger.info(f"Found {name} (S/N: {serial} "
                            f"FW: {fw_version} on USB {usb_type})")
                assert "D435" in name, "Only support D435 currently"
                rs_devices[serial] = d
        logger.info(f"Found {len(rs_devices)} devices")

    if device_sn is None:
        return list(rs_devices.values())
    elif isinstance(device_sn, str):
        return rs_devices[device_sn]
    else:
        return [rs_devices[sn] for sn in device_sn]


class RSDevice:
    """Only meant for D435"""

    def __init__(self, device: rs.device, config=None, preset="Default",
                 depth_option_kwargs={}, color_option_kwargs={}):
        self.logger = get_logger("RSDevice")

        self.device = device
        self.config = self.get_default_config(config)
        self.align = rs.align(rs.stream.color)

        self.pipeline = None
        self.pipeline_profile = None
        self.intrinsic_matrix = None
        self.last_frame_num = None

        self.load_depth_sensor_preset(preset)
        self.init_sensor(depth_option_kwargs, color_option_kwargs)

    def __repr__(self):
        name = self.device.get_info(rs.camera_info.name)
        serial = self.serial_number

        return f"<RSDevice: {name} (S/N: {serial})>"

    def init_sensor(self, depth_option_kwargs, color_option_kwargs):
        depth_sensor = self.depth_sensor
        for key, value in depth_option_kwargs.items():
            depth_sensor.set_option(key, value)
            self.logger.info(f'Setting Depth "{key}" to {value}')

        color_sensor = self.color_sensor
        for key, value in color_option_kwargs.items():
            color_sensor.set_option(key, value)
            self.logger.info(f'Setting Color "{key}" to {value}')

    @staticmethod
    def get_default_config(config: rs.config = None) -> rs.config:
        if config is not None:
            assert isinstance(config, rs.config)
            return config

        config = rs.config()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
        return config

    @property
    def serial_number(self) -> str:
        return self.device.get_info(rs.camera_info.serial_number)

    @property
    def color_sensor(self) -> rs.color_sensor:
        return self.device.first_color_sensor()

    @property
    def depth_sensor(self) -> rs.depth_sensor:
        return self.device.first_depth_sensor()

    @property
    def depth_sensor_presets(self) -> List[str]:
        depth_sensor = self.depth_sensor

        presets = []
        for i in range(10):
            preset = depth_sensor.get_option_value_description(
                rs.option.visual_preset, i
            )
            if preset == "UNKNOWN":
                break
            presets.append(preset)
        return presets

    def load_depth_sensor_preset(self, preset="Default"):
        if preset not in (presets := self.depth_sensor_presets):
            raise ValueError(f"No preset named {preset}. "
                             f"Available presets {presets}")

        self.depth_sensor.set_option(rs.option.visual_preset,
                                     presets.index(preset))
        self.logger.info(f'Loaded "{preset}" preset for {self}')

    def start(self):
        self.pipeline = rs.pipeline()

        self.config.enable_device(self.serial_number)
        self.pipeline_profile = self.pipeline.start(self.config)

        for _ in range(20):  # wait for white balance to stabilize
            self.pipeline.wait_for_frames()

        streams = self.pipeline_profile.get_streams()
        self.logger.info(f"Started device {self} with {len(streams)} streams")
        for stream in streams:
            self.logger.info(f"{stream}")

        # with rs.align, camera intrinsics is color sensor intrinsics
        stream_profile = self.pipeline_profile.get_stream(rs.stream.color)
        intrinsics = stream_profile.as_video_stream_profile().intrinsics
        self.intrinsic_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                          [0, intrinsics.fy, intrinsics.ppy],
                                          [0, 0, 1]], dtype=np.float32)

    def get_intrinsic_matrix(self) -> np.ndarray:
        """Returns a 3x3 camera intrinsics matrix"""
        return self.intrinsic_matrix

    def wait_for_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return color_image: color image, [H, W, 3] np.uint8 array
        :return depth_image: depth image, [H, W] np.uint16 array
        """
        assert self.pipeline is not None, "Device is not started"

        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        self.last_frame_num = frames.get_frame_number()
        # self.logger.info(f"Received frame #{self.last_frame_num}")

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        depth_image = np.asanyarray(depth_frame.get_data()).copy()
        color_image = np.asanyarray(color_frame.get_data()).copy()

        return color_image, depth_image

    def stop(self):
        self.pipeline.stop()
        self.pipeline = None
        self.pipeline_profile = None
        self.intrinsic_matrix = None
        self.last_frame_num = None
        self.logger.info(f"Stopped device {self}")


class RealSenseAPI:
    def __init__(self, **kwargs):
        self.logger = get_logger("RealSenseAPI")

        self._connected_devices = self.load_connected_devices(**kwargs)
        self._enabled_devices = []

        self.enable_all_devices()

    def __del__(self):
        self.disable_all_devices()

    def load_connected_devices(self, device_sn: Optional[List[str]] = None,
                               **kwargs) -> List[RSDevice]:
        """Return list of RSDevice
        :param device_sn: list of serial numbers of devices to load.
                          If not None, only load those devices in exact order.
        """
        if isinstance(device_sn, str):
            device_sn = [device_sn]
        devices = get_connected_rs_devices(device_sn)

        devices = [RSDevice(d, **kwargs) for d in devices]

        self.logger.info(f"Loading {len(devices)} devices")
        return devices

    def enable_all_devices(self):
        for device in self._connected_devices:
            device.start()
            self._enabled_devices.append(device)

    def capture(self):
        """Capture data from all _enabled_devices.
        If n_cam == 1, first dimension is squeezed.
        :return color_image: color image, [n_cam, H, W, 3] np.uint8 array
        :return depth_image: depth image, [n_cam, H, W] np.uint16 array
        """
        color_images = []
        depth_images = []
        for device in self._enabled_devices:
            color_image, depth_image = device.wait_for_frames()
            color_images.append(color_image)
            depth_images.append(depth_image)

        if len(self._enabled_devices) == 1:
            color_images = color_images[0]
            depth_images = depth_images[0]
        else:
            color_images = np.stack(color_images)
            depth_images = np.stack(depth_images)

        return color_images, depth_images

    def disable_all_devices(self):
        for i in range(len(self._enabled_devices)):
            device = self._enabled_devices.pop()
            device.stop()
