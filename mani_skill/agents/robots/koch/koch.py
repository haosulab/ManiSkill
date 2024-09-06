from copy import deepcopy
from typing import List

import numpy as np
import sapien
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import vectorize_pose


@register_agent()
class Koch(BaseAgent):
    uid = "koch"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/koch/Follower_Arm.urdf"
    urdf_config = dict()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # NOTE: Controller is temporary - doesn't resemble real robot
    @property
    def _controller_configs(self):
        joint_delta_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            -0.1,
            0.1,
            stiffness=1e4,
            damping=1e3,
            force_limit=200,
            use_delta=True,
        )

        controller_configs = dict(
            pd_joint_delta_pos=joint_delta_pos,
        )
        return deepcopy_dict(controller_configs)