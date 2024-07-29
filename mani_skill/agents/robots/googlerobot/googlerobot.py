import numpy as np
import sapien

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


# TODO (stao) (xuanlin): Add mobile base, model it properly based on real2sim
@register_agent(asset_download_ids=["googlerobot"])
class GoogleRobot(BaseAgent):
    uid = "googlerobot"
    urdf_path = (
        f"{ASSET_DIR}/robots/googlerobot/google_robot_meta_sim_fix_fingertip.urdf"
    )
    urdf_config = dict()

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="overhead_camera",
                pose=sapien.Pose([0, 0, 0], [0.5, 0.5, -0.5, 0.5]),
                width=640,
                height=512,
                entity_uid="link_camera",
                intrinsic=np.array([[425.0, 0, 305.0], [0, 413.1, 233.0], [0, 0, 1]]),
            )
        ]
