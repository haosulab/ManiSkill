import numpy as np
import sapien

import torch 
from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

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
    
    # def _after_loading_articulation(self):
    #     super()._after_loading_articulation()
    #     return 
    #     self.finger1_link = self.robot.links_map["Fixed_Jaw"]
    #     self.finger2_link = self.robot.links_map["Moving_Jaw"]
    #     self.finger1_tip = self.robot.links_map["Fixed_Jaw_tip"]
    #     self.finger2_tip = self.robot.links_map["Moving_Jaw_tip"]

    
    # @property
    # def tcp_pos(self):
    #     # computes the tool center point as the mid point between the the fixed and moving jaw's tips
    #     return (self.finger1_tip.pose.p + self.finger2_tip.pose.p) / 2

    # @property
    # def tcp_pose(self):
    #     return Pose.create_from_pq(self.tcp_pos, self.finger1_link.pose.q)


    # def is_grasping(self, object: Actor, min_force=0.5, max_angle=110):
    #     return False

    # def is_static(self, threshold=0.2):
    #     qvel = self.robot.get_qvel()[
    #         :, :-2
    #     ]  # exclude the gripper joint and gripper rotation joint.
    #     return torch.max(torch.abs(qvel), 1)[0] <= threshold
