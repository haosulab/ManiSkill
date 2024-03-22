import numpy as np

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs import Pose

from .panda import Panda


@register_agent()
class PandaRealSensed435(Panda):
    """Panda arm robot with the real sense camera attached to gripper"""

    uid = "panda_realsensed435"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v3.urdf"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="hand_camera",
                pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=sapien_utils.get_obj_by_name(self.robot.links, "camera_link"),
            )
        ]
