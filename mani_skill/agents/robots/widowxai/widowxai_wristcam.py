import numpy as np
import sapien

from mani_skill import ASSET_DIR
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

from .widowxai import WidowXAI


@register_agent(asset_download_ids=["widowxai"])
class WidowXAIWristCam(WidowXAI):
    """WidowX AI robot with a Intel Realsense D405 mounted on the gripper"""

    uid = "widowxai_wristcam"
    urdf_path = f"{ASSET_DIR}/robots/widowxai/wxai_follower.urdf"

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
