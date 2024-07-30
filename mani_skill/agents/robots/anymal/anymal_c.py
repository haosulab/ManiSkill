# TODO (stao): Anymal may not be modelled correctly or efficiently at the moment
import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.articulation import Articulation


@register_agent(asset_download_ids=["anymal_c"])
class ANYmalC(BaseAgent):
    uid = "anymal_c"
    urdf_path = f"{ASSET_DIR}/robots/anymal_c/urdf/anymal.urdf"
    urdf_config = dict(
        _materials=dict(
            foot=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            LF_FOOT=dict(material="foot", patch_radius=0.1, min_patch_radius=0.1),
            LH_FOOT=dict(material="foot", patch_radius=0.1, min_patch_radius=0.1),
            RF_FOOT=dict(material="foot", patch_radius=0.1, min_patch_radius=0.1),
            RH_FOOT=dict(material="foot", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )
    fix_root_link = False
    disable_self_collisions = True

    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.545]),
            qpos=np.array(
                [0.03, -0.03, 0.03, -0.03, 0.4, 0.4, -0.4, -0.4, -0.8, -0.8, 0.8, 0.8]
            ),
        )
    )

    joint_names = [
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        self.stiffness = 80.0
        self.damping = 2.0
        self.force_limit = 100
        # delta action scale for Omni Isaac Gym Envs is self.dt * self.action_scale = 1/60 * 13.5. NOTE that their self.dt value is not the same as the actual DT used in sim...., they use default of 1/100
        pd_joint_delta_pos = PDJointPosControllerConfig(
            self.joint_names,
            -0.225,
            0.225,
            self.stiffness,
            self.damping,
            self.force_limit,
            normalize_action=True,
            use_delta=True,
        )
        pd_joint_pos = PDJointPosControllerConfig(
            self.joint_names,
            None,
            None,
            self.stiffness,
            self.damping,
            self.force_limit,
            normalize_action=False,
            use_delta=False,
        )
        # TODO (stao): For quadrupeds perhaps we disable gravity for all links except the root?
        controller_configs = dict(
            pd_joint_delta_pos=dict(
                body=pd_joint_delta_pos, balance_passive_force=False
            ),
            pd_joint_pos=dict(body=pd_joint_pos, balance_passive_force=False),
        )
        return controller_configs

    def _after_init(self):
        # disable gravity / compensate gravity automatically in all links but the root one
        for link in self.robot.links[1:]:
            link.disable_gravity = True

    def is_standing(self, ground_height=0):
        """This quadruped is considered standing if it is face up and body is at least 0.35m off the ground"""
        target_q = torch.tensor([1, 0, 0, 0], device=self.device)
        inner_prod = (self.robot.pose.q * target_q).sum(axis=1)
        # angle_diff = 1 - (inner_prod ** 2) # computes a distance from 0 to 1 between 2 quaternions
        angle_diff = torch.arccos(
            2 * (inner_prod**2) - 1
        )  # computes an angle between 2 quaternions
        # about 20 degrees
        aligned = angle_diff < 0.349
        high_enough = self.robot.pose.p[:, 2] > 0.35 + ground_height
        return aligned & high_enough

    def is_fallen(self):
        """This quadruped is considered fallen if its body contacts the ground"""
        forces = self.robot.get_net_contact_forces(["base"])
        return torch.norm(forces, dim=-1).max(-1).values > 1
