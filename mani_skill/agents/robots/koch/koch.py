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
    # mjcf_path = f"{PACKAGE_ASSET_DIR}/robots/koch/low_cost_robot.xml"
    # urdf_config = dict()
    #load_multiple_collisions = True
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/koch/Follower_Arm.urdf"
    urdf_config = dict(
        # _materials=dict(
        #     pad=dict(static_friction=0.7, dynamic_friction=0.7, restitution=0.0)
        # ),
        # link={
        #     "joint4-pad": dict(
        #         material="pad", patch_radius=0.1, min_patch_radius=0.1
        #     ),
        #     "joint5-pad": dict(
        #         material="pad", patch_radius=0.1, min_patch_radius=0.1
        #     ),
        # },
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    # NOTE: S
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