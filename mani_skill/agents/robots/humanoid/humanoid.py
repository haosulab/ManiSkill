import numpy as np
import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


@register_agent()
class Humanoid(BaseAgent):
    uid = "humanoid"
    mjcf_path = f"{PACKAGE_ASSET_DIR}/robots/humanoid/humanoid.xml"
    urdf_config = dict()
    fix_root_link = False  # False as there is a freejoint on the root body

    keyframes = dict(
        squat=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.12,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.25,
                    -0.25,
                    -0.25,
                    -0.25,
                    -0.5,
                    -0.5,
                    -1.3,
                    -1.3,
                    -0.8,
                    -0.8,
                    0.0,
                    0.0,
                ]
            ),
            pose=sapien.Pose(p=[0, 0, -0.375]),
        )
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        pd_joint_pos = PDJointPosControllerConfig(
            [x.name for x in self.robot.active_joints],
            lower=None,
            upper=None,
            stiffness=100,
            damping=10,
            normalize_action=False,
        )
        pd_joint_delta_pos = PDJointPosControllerConfig(
            [j.name for j in self.robot.active_joints],
            -1,
            1,
            damping=5,
            stiffness=20,
            force_limit=100,
            use_delta=True,
        )
        return deepcopy_dict(
            dict(
                pd_joint_pos=dict(body=pd_joint_pos, balance_passive_force=False),
                pd_joint_delta_pos=dict(
                    body=pd_joint_delta_pos, balance_passive_force=False
                ),
            )
        )
