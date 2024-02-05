import torch

from mani_skill2 import PACKAGE_ASSET_DIR
from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.controllers import *


class ANYmalC(BaseAgent):
    uid = "anymal-c"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/anymal-c/urdf/anymal.urdf"
    urdf_config = dict()

    def __init__(self, *args, **kwargs):
        self.joint_names = [
            "LF_HAA",
            "RF_HAA",
            "LH_HAA",
            "RH_HAA",
            "LF_HFE",
            "RF_HFE",
            "LH_HFE",
            "RH_HFE",
            "LF_KFE",
            "RF_KFE",
            "LH_KFE",
            "RH_KFE",
        ]
        super().__init__(*args, fix_root_link=False, **kwargs)

    @property
    def controller_configs(self):
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100
        # import ipdb;ipdb.set_trace()
        pd_joint_delta_pos = PDJointPosControllerConfig(
            self.joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=True,
            use_delta=True,
        )
        pd_joint_pos = PDJointPosControllerConfig(
            self.joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
            use_delta=False,
        )
        controller_configs = dict(
            pd_joint_delta_pos=pd_joint_delta_pos, pd_joint_pos=pd_joint_pos
        )
        return controller_configs

    def _after_init(self):
        pass

    def is_standing(self, q_thresh=10):
        """This quadruped is considered standing if it is face up and body is at least 0.3m off the ground"""
        target_q = torch.tensor([1, 0, 0, 0], device=self.device)
        inner_prod = (self.robot.pose.q * target_q).sum(axis=1)
        # angle_diff = 1 - (inner_prod ** 2) # computes a distance from 0 to 1 between 2 quaternions
        angle_diff = torch.arccos(
            2 * (inner_prod**2) - 1
        )  # computes an angle between 2 quaternions
        # < 10 degree
        aligned = angle_diff < 0.17453292519943295
        return aligned

    sensor_configs = []
