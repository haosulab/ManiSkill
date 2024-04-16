from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


@register_agent()
class Humanoid(BaseAgent):
    uid = "humanoid"
    mjcf_path = f"{PACKAGE_ASSET_DIR}/robots/humanoid/humanoid.xml"
    urdf_config = dict()
    fix_root_link = False  # False as there is a freejoint on the root body

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        pd_joint_delta_pos = PDJointPosControllerConfig(
            [j.name for j in self.robot.active_joints],
            -1,
            1,
            damping=5,
            stiffness=20,
            force_limit=100,
            use_delta=True,
        )
        return dict(pd_joint_delta_pos=pd_joint_delta_pos)

    def _after_init(self):
        pass
