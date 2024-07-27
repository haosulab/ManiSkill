import numpy as np
import sapien

from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent


@register_agent(asset_download_ids=["ur10e"])
class UR10e(BaseAgent):
    uid = "ur_10e"
    mjcf_path = f"{ASSET_DIR}/robots/ur10e/ur10e.xml"
    urdf_config = dict()

    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0]),
            qpos=np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]),
        )
    )

    @property
    def _controller_configs(
        self,
    ):
        return dict(
            pd_joint_pos=PDJointPosControllerConfig(
                [x.name for x in self.robot.active_joints],
                lower=None,
                upper=None,
                stiffness=1000,
                damping=100,
                normalize_action=False,
            ),
            pd_joint_delta_pos=PDJointPosControllerConfig(
                [x.name for x in self.robot.active_joints],
                lower=-0.1,
                upper=0.1,
                stiffness=1e4,
                damping=1e3,
                normalize_action=True,
                use_delta=True,
            ),
        )
