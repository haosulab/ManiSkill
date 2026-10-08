import numpy as np
import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig

from .so_100 import SO100


@register_agent()
class SO100WristCam(SO100):
    """SO100 arm robot with a centered wrist camera mounted above the wrist joint (like the
    official SO-ARM100 wrist mount cameras). The camera is mounted on the ``camera_link`` of
    the so100.urdf and looks down past the gripper at the workspace."""

    uid = "so100_wristcam"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/so100/so100.urdf"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="wrist_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            )
        ]
