import numpy as np
import sapien
import pytest

from mani_skill.sensors.camera import CameraConfig, parse_sensor_configs, update_sensor_configs_from_dict
from mani_skill.sensors.depth_camera import StereoDepthCameraConfig
from mani_skill.utils.structs.pose import Pose


def make_camera_config(**kwargs):
    return CameraConfig(
        uid="base_cam",
        pose=Pose.create(sapien.Pose([0, 0, 1], [1, 0, 0, 0])),
        width=64,
        height=48,
        fov=1.0,
        **kwargs,
    )


def test_stereo_depth_camera_config_from_camera_config():
    cfg = StereoDepthCameraConfig.fromCameraConfig(make_camera_config())
    assert isinstance(cfg, StereoDepthCameraConfig)
    assert isinstance(cfg, CameraConfig)
    assert cfg.uid == "base_cam"
    assert cfg.width == 64 and cfg.height == 48
    assert cfg.min_depth == 0.05 and cfg.max_depth == 10.0 and cfg.model == "D435"
    # fov is converted into an intrinsic matrix matching the CameraConfig convention
    assert cfg.rgb_resolution == (64, 48)
    fy = (48 / 2) / np.tan(1.0 / 2)
    assert cfg.rgb_intrinsic.shape == (3, 3)
    assert cfg.rgb_intrinsic[0, 0] == pytest.approx(fy)
    assert cfg.rgb_intrinsic[0, 2] == 32 and cfg.rgb_intrinsic[1, 2] == 24


def test_stereo_depth_camera_config_explicit_fields():
    cfg = StereoDepthCameraConfig(
        uid="cam",
        pose=Pose.create(sapien.Pose([0, 0, 1], [1, 0, 0, 0])),
        width=64,
        height=48,
        fov=1.0,
        min_depth=0.1,
        max_depth=5.0,
        model="D415",
    )
    assert cfg.min_depth == 0.1 and cfg.max_depth == 5.0 and cfg.model == "D415"
    with pytest.raises(AssertionError):
        StereoDepthCameraConfig(
            uid="cam",
            pose=Pose.create(sapien.Pose([0, 0, 1], [1, 0, 0, 0])),
            width=64,
            height=48,
            fov=1.0,
            model="D999",
        )


def test_parse_sensor_configs_stereo():
    cfgs = parse_sensor_configs(
        StereoDepthCameraConfig.fromCameraConfig(make_camera_config())
    )
    assert isinstance(cfgs["base_cam"], StereoDepthCameraConfig)


def test_update_sensor_configs_from_dict_use_stereo_depth():
    sensor_configs = {"base_cam": make_camera_config()}
    update_sensor_configs_from_dict(sensor_configs, {"use_stereo_depth": True})
    assert isinstance(sensor_configs["base_cam"], StereoDepthCameraConfig)


def test_update_sensor_configs_from_dict_per_camera_use_stereo_depth():
    sensor_configs = {
        "base_cam": make_camera_config(),
        "other_cam": CameraConfig(
            uid="other_cam",
            pose=Pose.create(sapien.Pose([0, 0, 1], [1, 0, 0, 0])),
            width=32,
            height=32,
            fov=1.0,
        ),
    }
    update_sensor_configs_from_dict(
        sensor_configs, {"base_cam": {"use_stereo_depth": True}}
    )
    assert isinstance(sensor_configs["base_cam"], StereoDepthCameraConfig)
    assert not isinstance(sensor_configs["other_cam"], StereoDepthCameraConfig)
