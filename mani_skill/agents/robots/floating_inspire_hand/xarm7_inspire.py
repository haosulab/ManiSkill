from copy import deepcopy
from typing import Dict, Tuple

import sapien.physx as physx
import torch
from transforms3d import euler

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor


@register_agent()
class Xarm7InspireRightHand(BaseAgent):
    uid = "xarm7_inspire"
    urdf_path = (
        f"{PACKAGE_ASSET_DIR}/robots/xarm7_inspire_hand/xarm7_inspire_right_hand.urdf"
    )
    urdf_config = dict(
        _materials=dict(
            front_finger=dict(static_friction=5.0, dynamic_friction=4, restitution=0.0),
            mid_part=dict(static_friction=2.0, dynamic_friction=1.5, restitution=0.0),
        ),
        link=dict(
            right_link11=dict(
                material="front_finger", patch_radius=0.05, min_patch_radius=0.04
            ),
            right_link22=dict(
                material="front_finger", patch_radius=0.05, min_patch_radius=0.04
            ),
            right_link33=dict(
                material="front_finger", patch_radius=0.05, min_patch_radius=0.04
            ),
            right_link44=dict(
                material="front_finger", patch_radius=0.05, min_patch_radius=0.04
            ),
            right_link53=dict(
                material="front_finger", patch_radius=0.05, min_patch_radius=0.04
            ),
            right_link52=dict(
                material="mid_part", patch_radius=0.05, min_patch_radius=0.04
            ),
            right_link51=dict(
                material="mid_part", patch_radius=0.05, min_patch_radius=0.04
            ),
            right_link5=dict(
                material="mid_part", patch_radius=0.05, min_patch_radius=0.04
            ),
            right_link4=dict(
                material="mid_part", patch_radius=0.05, min_patch_radius=0.04
            ),
            right_link3=dict(
                material="mid_part", patch_radius=0.05, min_patch_radius=0.04
            ),
            right_link2=dict(
                material="mid_part", patch_radius=0.05, min_patch_radius=0.04
            ),
            right_link1=dict(
                material="mid_part", patch_radius=0.05, min_patch_radius=0.04
            ),
            right_base_link=dict(
                material="mid_part", patch_radius=0.05, min_patch_radius=0.04
            ),
        ),
    )
    disable_self_collisions = True

    def __init__(self, *args, **kwargs):
        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.arm_stiffness = 1e4
        self.arm_damping = 1e3
        self.arm_force_limit = 500

        self.hand_joint_names = [
            "right_joint1",
            "right_joint2",
            "right_joint3",
            "right_joint4",
            "right_joint5",
            "right_joint51",
            "right_joint11",
            "right_joint22",
            "right_joint33",
            "right_joint44",
            "right_joint52",
            "right_joint53",
        ]
        self.hand_stiffness = 1e3
        self.hand_damping = 1e2
        self.hand_force_limit = 15

        self.ee_link_name = "right_base_link"
        super().__init__(*args, **kwargs)

    # @property
    # def _sensor_configs(self):
    #     return [
    #         CameraConfig(
    #             uid="hand_camera",
    #             pose=Pose.create_from_pq([0.0, -0.08, 0.02], euler.euler2quat(0, -np.pi / 2, -np.pi / 2)),
    #             width=128,
    #             height=128,
    #             fov=np.pi / 2,
    #             near=0.01,
    #             far=100,
    #             mount=sapien_utils.get_obj_by_name(self.robot.get_links(), "right_base_link"),
    #         )
    #     ]

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            # frame="ee_align",
            # use_delta=False,
            # normalize_action=False,
        )

        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # -------------------------------------------------------------------------- #
        # Hand
        # -------------------------------------------------------------------------- #

        hand_mimic_delta_pos = PDJointPosMimicControllerConfig(
            self.hand_joint_names,
            -0.1,
            0.1,
            self.hand_stiffness,
            self.hand_damping,
            self.hand_force_limit,
            drive_mode="acceleration",
            use_delta=True,
            multiplier=dict(
                right_joint11=0.9288214702450408,
                right_joint22=0.9487790351399642,
                right_joint33=1.0414634146341466,
                right_joint44=1.0076741440377803,
                right_joint52=2.252847380410023,
                right_joint53=2.7175398633257406,
            ),
            offset=dict(
                right_joint11=-0.3029918319719953,
                right_joint22=-0.055416914830256125,
                right_joint33=0.2798292682926833,
                right_joint44=0.003002361275088461,
                right_joint52=-0.43612072892938497,
                right_joint53=-0.4544236902050113,
            ),
            mimic={
                "right_joint11": "right_joint1",
                "right_joint22": "right_joint2",
                "right_joint33": "right_joint3",
                "right_joint44": "right_joint4",
                "right_joint52": "right_joint51",
                "right_joint53": "right_joint51",
            },
        )
        hand_mimic_pos = deepcopy(hand_mimic_delta_pos)
        hand_mimic_pos.use_delta = False

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=hand_mimic_delta_pos
            ),
            # pd_joint_mimic_delta_pos=dict(
            #     arm=arm_pd_joint_delta_pos, gripper=hand_mimic_delta_pos
            # ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=hand_mimic_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=hand_mimic_delta_pos
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=hand_mimic_delta_pos
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def reset(self):
        init_qpos = torch.zeros_like(self.robot.get_qpos())
        n = init_qpos.shape[0]
        target_qpos = torch.tensor(
            [
                -9.0390e-07,
                -2.6845e-07,
                -1.9546e-06,
                2.0661e-07,
                1.1883e-05,
                -2.4855e-07,
                1.3961e-06,
                2.5300e-01,
                1.0900e-01,
                -2.0000e-02,
                1.3000e-01,
                -3.2900e-01,
                -6.8000e-02,
                4.8000e-02,
                2.5900e-01,
                1.3400e-01,
                2.1800e-01,
                5.5000e-02,
                1.3800e-01,
            ],
            device=init_qpos.device,
        )
        init_qpos = target_qpos.repeat(n, 1)
        super().reset(init_qpos=init_qpos)

    def _after_init(self):
        hand_front_link_names = [
            "right_link1",
            "right_link2",
            "right_link3",
            "right_link4",
            "right_link5",
        ]
        self.hand_front_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(), hand_front_link_names
        )

        finger_tip_link_names = [
            "right_link11",
            "right_link22",
            "right_link33",
            "right_link44",
            "right_link53",
        ]
        self.finger_tip_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(), finger_tip_link_names
        )

        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

        self.hand_base = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_base_link"
        )

        self.queries: Dict[str, Tuple[physx.PhysxGpuContactQuery, Tuple[int]]] = dict()
