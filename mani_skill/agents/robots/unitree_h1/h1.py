from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


@register_agent()
class TemplateRobot(BaseAgent):
    uid = "unitree_h1"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/unitree_h1/urdf/h1.urdf"
    urdf_config = dict()
    fix_root_link = True

    @property
    def _controller_configs(self):
        raise NotImplementedError()

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="your_custom_camera_on_this_robot",
                p=[0.0464982, -0.0200011, 0.0360011],
                q=[0, 0.70710678, 0, 0.70710678],
                width=128,
                height=128,
                fov=1.57,
                near=0.01,
                far=100,
                entity_uid="your_mounted_camera",
            )
        ]

    def _after_init(self):
        pass
