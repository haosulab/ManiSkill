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
class AllegroHandRightFloating(BaseAgent):
    uid = "allegro_hand_right_floating"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/allegro/allegro_hand_right_floating.urdf"
    urdf_config = dict(
        _materials=dict(
            tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
        ),
        link={
            "link_3.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_7.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_11.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_15.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
        },
    )
    keyframes = dict(
        palm_side=Keyframe(
            qpos=np.zeros(16),
            pose=sapien.Pose([0, 0, 0.5], q=[1, 0, 0, 0]),
        ),
        palm_up=Keyframe(
            qpos=np.zeros(16),
            pose=sapien.Pose([0, 0, 0.5], q=[-0.707, 0, 0.707, 0]),
        ),
    )

    def __init__(self, *args, **kwargs):
        self.root_joint_names = [
            "root_x_axis_joint",
            "root_y_axis_joint",
            "root_z_axis_joint",
            "root_x_rot_joint",
            "root_y_rot_joint",
            "root_z_rot_joint",
          ]
        self.root_joint_stiffness = 5e3
        self.root_joint_damping = 5e2 
        self.root_joint_force_limit = 2e3 

        self.joint_names = [
            "joint_0.0",
            "joint_1.0",
            "joint_2.0",
            "joint_3.0",
            "joint_4.0",
            "joint_5.0",
            "joint_6.0",
            "joint_7.0",
            "joint_8.0",
            "joint_9.0",
            "joint_10.0",
            "joint_11.0",
            "joint_12.0",
            "joint_13.0",
            "joint_14.0",
            "joint_15.0",
        ]

        self.joint_stiffness = 2e4
        self.joint_damping = 300
        self.joint_force_limit = 5e1

        # Order: thumb finger, index finger, middle finger, ring finger
        self.tip_link_names = [
            "link_15.0_tip",
            "link_3.0_tip",
            "link_7.0_tip",
            "link_11.0_tip",
        ]

        self.palm_link_name = "palm"
        super().__init__(*args, **kwargs)

    def _after_init(self):
        self.tip_links: List[sapien.Entity] = sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.tip_link_names
        )
        self.palm_link: sapien.Entity = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.palm_link_name
        )

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        root_joint_pos = PDJointPosControllerConfig(
            self.root_joint_names,
            [-20, -20, -20, -6, -6, -6],
            [20, 20, 20, 6, 6, 6],
            self.root_joint_stiffness,
            self.root_joint_damping,
            self.root_joint_force_limit,
            normalize_action=False,
        )
        root_joint_delta_pos = PDJointPosControllerConfig(
            self.root_joint_names,
            -0.1,
            0.1,
            self.root_joint_stiffness,
            self.root_joint_damping,
            self.root_joint_force_limit,
            use_delta=True,
        )
        root_joint_target_delta_pos = deepcopy(root_joint_delta_pos)
        root_joint_target_delta_pos.use_target = True

        # -------------------------------------------------------------------------- #
        # Finger
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

        controller_configs = dict(
            pd_joint_delta_pos=dict(root=root_joint_delta_pos, hand=joint_delta_pos),
            pd_joint_pos=dict(root=root_joint_pos, hand=joint_pos),
            pd_joint_target_delta_pos=dict(root=root_joint_target_delta_pos, hand=joint_target_delta_pos)
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent.
        """
        obs = super().get_proprioception()
        obs.update(
            {
                "palm_pose": self.palm_pose,
                "tip_poses": self.tip_poses.reshape(-1, len(self.tip_links) * 7),
            }
        )

        return obs

    @property
    def tip_poses(self):
        """
        Get the tip pose for each of the finger, four fingers in total
        """
        tip_poses = [
            vectorize_pose(link.pose, device=self.device) for link in self.tip_links
        ]
        return torch.stack(tip_poses, dim=-2)

    @property
    def palm_pose(self):
        """
        Get the palm pose for allegro hand
        """
        return vectorize_pose(self.palm_link.pose, device=self.device)


@register_agent()
class AllegroHandLeftFloating(AllegroHandRightFloating):
    uid = "allegro_hand_left_floating"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/allegro/allegro_hand_left.urdf"
