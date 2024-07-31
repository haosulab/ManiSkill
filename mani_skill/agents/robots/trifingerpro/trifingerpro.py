from copy import deepcopy
from typing import List

import sapien
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.agents.utils import get_active_joint_indices
from mani_skill.utils.sapien_utils import get_objs_by_names
from mani_skill.utils.structs.pose import vectorize_pose


@register_agent()
class TriFingerPro(BaseAgent):
    """
    Modified from https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/trifinger.py

    """

    uid = "trifingerpro"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/trifinger/trifingerpro.urdf"
    urdf_config = dict(
        _materials=dict(
            tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
        ),
        link=dict(
            finger_tip_link_0=dict(material="tip"),
            finger_tip_link_120=dict(material="tip"),
            finger_tip_link_240=dict(material="tip"),
        ),
    )
    sensor_configs = {}

    def __init__(self, *args, **kwargs):
        self.joint_names = [
            # "base_to_upper_holder_joint",
            # "finger_upper_visuals_joint_0",
            # "finger_middle_visuals_joint_0",
            # "finger_lower_to_tip_joint_0",
            "finger_base_to_upper_joint_0",
            "finger_upper_to_middle_joint_0",
            "finger_middle_to_lower_joint_0",
            # "finger_upper_visuals_joint_120",
            # "finger_middle_visuals_joint_120",
            # "finger_lower_to_tip_joint_120",
            "finger_base_to_upper_joint_120",
            "finger_upper_to_middle_joint_120",
            "finger_middle_to_lower_joint_120",
            # "finger_upper_visuals_joint_240",
            # "finger_middle_visuals_joint_240",
            # "finger_lower_to_tip_joint_240",
            "finger_base_to_upper_joint_240",
            "finger_upper_to_middle_joint_240",
            "finger_middle_to_lower_joint_240",
            # "holder_to_finger_0",
            # "holder_to_finger_120",
            # "holder_to_finger_240",
        ]

        self.joint_stiffness = 1e2
        self.joint_damping = 1e1
        self.joint_force_limit = 2e1
        self.tip_link_names = [
            "finger_tip_link_0",
            "finger_tip_link_120",
            "finger_tip_link_240",
        ]
        self.root_joint_names = [
            "finger_base_to_upper_joint_0",
            "finger_base_to_upper_joint_120",
            "finger_base_to_upper_joint_240",
        ]

        super().__init__(*args, **kwargs)

    def _after_init(self):
        self.tip_links: List[sapien.Entity] = get_objs_by_names(
            self.robot.get_links(), self.tip_link_names
        )
        self.root_joints = [
            self.robot.find_joint_by_name(n) for n in self.root_joint_names
        ]
        self.root_joint_indices = get_active_joint_indices(
            self.robot, self.root_joint_names
        )

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        joint_pos = PDJointPosControllerConfig(
            self.joint_names,
            None,
            None,
            self.joint_stiffness,
            self.joint_damping,
            self.joint_force_limit,
            normalize_action=False,
        )
        joint_delta_pos = PDJointPosControllerConfig(
            self.joint_names,
            -0.1,
            0.1,
            self.joint_stiffness,
            self.joint_damping,
            self.joint_force_limit,
            use_delta=True,
        )
        joint_target_delta_pos = deepcopy(joint_delta_pos)
        joint_target_delta_pos.use_target = True

        # PD joint velocity
        pd_joint_vel = PDJointVelControllerConfig(
            self.joint_names,
            -1.0,
            1.0,
            self.joint_damping,  # this might need to be tuned separately
            self.joint_force_limit,
        )

        # PD joint position and velocity
        joint_pos_vel = PDJointPosVelControllerConfig(
            self.joint_names,
            None,
            None,
            self.joint_stiffness,
            self.joint_damping,
            self.joint_force_limit,
            normalize_action=False,
        )
        joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.joint_names,
            -0.1,
            0.1,
            self.joint_stiffness,
            self.joint_damping,
            self.joint_force_limit,
            use_delta=True,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(joint=joint_delta_pos),
            pd_joint_pos=dict(joint=joint_pos),
            pd_joint_target_delta_pos=dict(joint=joint_target_delta_pos),
            # Caution to use the following controllers
            pd_joint_vel=dict(joint=pd_joint_vel),
            pd_joint_pos_vel=dict(joint=joint_pos_vel),
            pd_joint_delta_pos_vel=dict(joint=joint_delta_pos_vel),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent.
        """
        obs = super().get_proprioception()
        obs.update({"tip_poses": self.tip_poses.view(-1, 21)})
        obs.update({"tip_velocities": self.tip_velocities().view(-1, 9)})
        return obs

    @property
    def tip_poses(self):
        """
        Get the tip pose for each of the finger, three fingers in total
        """
        tip_poses = [
            vectorize_pose(link.pose, device=self.device) for link in self.tip_links
        ]
        return torch.stack(tip_poses, dim=-1)

    # @property
    def tip_velocities(self):
        """
        Get the tip velocity for each of the finger, three fingers in total
        """
        tip_velocities = [link.linear_velocity for link in self.tip_links]
        return torch.stack(tip_velocities, dim=-1)
