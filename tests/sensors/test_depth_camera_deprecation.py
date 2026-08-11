import warnings

import sapien

from mani_skill.sensors.camera import CameraConfig, update_sensor_configs_from_dict
from mani_skill.sensors.depth_camera import (
    _STEREO_DEPTH_CAMERA_DEPRECATION_MESSAGE,
    StereoDepthCameraConfig,
)


def make_camera_config():
    return CameraConfig(
        uid="camera",
        pose=sapien.Pose(),
        width=64,
        height=64,
        fov=1.0,
    )


def test_stereo_depth_camera_config_warns_on_direct_use():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        StereoDepthCameraConfig(
            uid="camera",
            pose=sapien.Pose(),
            width=64,
            height=64,
            fov=1.0,
        )

    assert len(caught) == 1
    assert caught[0].category is DeprecationWarning
    assert str(caught[0].message) == _STEREO_DEPTH_CAMERA_DEPRECATION_MESSAGE
    assert caught[0].filename == __file__


def test_use_stereo_depth_override_warns():
    sensor_configs = {"camera": make_camera_config()}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        update_sensor_configs_from_dict(sensor_configs, {"use_stereo_depth": True})

    assert len(caught) == 1
    assert caught[0].category is DeprecationWarning
    assert str(caught[0].message) == _STEREO_DEPTH_CAMERA_DEPRECATION_MESSAGE
    assert isinstance(sensor_configs["camera"], StereoDepthCameraConfig)
