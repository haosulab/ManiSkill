from mani_skill2 import PACKAGE_ASSET_DIR
from mani_skill2.sensors.camera import CameraConfig

from .panda import Panda


class PandaRealSensed435(Panda):
    uid = "panda_realsensed435"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v3.urdf"
    sensor_configs = [
        CameraConfig(
            uid="hand_camera",
            p=[0, 0, 0],
            q=[1, 0, 0, 0],
            width=128,
            height=128,
            fov=1.57,
            near=0.01,
            far=10,
            entity_uid="camera_link",
        )
    ]
