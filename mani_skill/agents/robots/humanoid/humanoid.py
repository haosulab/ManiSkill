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
        # pd_joint_pos = PDJointPosControllerConfig(
        #     [x.name for x in self.robot.active_joints],
        #     lower=None,
        #     upper=None,
        #     stiffness=100,
        #     damping=10,
        #     normalize_action=False,
        # )
        pd_joint_pos = PDJointPosControllerConfig(
            [x.name for x in self.robot.active_joints],
            lower=None,
            upper=None,
            stiffness=1000,
            damping=10,
            normalize_action=False,
        )

        # pd_joint_delta_pos = PDJointPosControllerConfig(  # og sac good
        #     [j.name for j in self.robot.active_joints],
        #     -1,
        #     1,
        #     damping=10,
        #     stiffness=200,
        #     force_limit=200,
        #     use_delta=True,
        # )

        # pd_joint_delta_pos = PDJointPosControllerConfig(  # good for walk
        #     [j.name for j in self.robot.active_joints],
        #     -0.5,
        #     0.5,
        #     damping=10,
        #     stiffness=200,
        #     #force_limit=200,
        #     use_delta=True,
        # )

        # pd_joint_delta_pos = PDJointPosControllerConfig(  # good for walk
        #     [j.name for j in self.robot.active_joints],
        #     -1,
        #     1,
        #     damping=10,
        #     stiffness=200,
        #     #force_limit=200,
        #     use_delta=True,
        # )

        # return deepcopy_dict(
        #     dict(
        #         pd_joint_pos=dict(body=pd_joint_pos, balance_passive_force=False),
        #         pd_joint_delta_pos=dict(
        #             body=pd_joint_delta_pos, balance_passive_force=False
        #         ),
        #     )
        # )

        pd_joint_delta_pos = dict()

        # pd_joint_delta_pos.update(abdomen_y = PDJointPosControllerConfig(  # good for walk
        #     ["abdomen_y"],-max_delta,max_delta,damping=base_stiff*40/20,stiffness=base_stiff*40,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(abdomen_z = PDJointPosControllerConfig(  # good for walk
        #     ["abdomen_z"],-max_delta,max_delta,damping=base_stiff*40/20,stiffness=base_stiff*40,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(abdomen_x = PDJointPosControllerConfig(  # good for walk
        #     ["abdomen_x"],-max_delta,max_delta,damping=base_stiff*40/20,stiffness=base_stiff*40,use_delta=True,
        # ))

        # pd_joint_delta_pos.update(right_hip_x = PDJointPosControllerConfig(  # good for walk
        #     ["right_hip_x"],-max_delta,max_delta,damping=base_stiff*40/20,stiffness=base_stiff*40,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_hip_z = PDJointPosControllerConfig(  # good for walk
        #     ["right_hip_z"],-max_delta,max_delta,damping=base_stiff*40/20,stiffness=base_stiff*40,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_hip_y = PDJointPosControllerConfig(  # good for walk
        #     ["right_hip_y"],-max_delta,max_delta,damping=base_stiff*120/20,stiffness=base_stiff*120,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_knee = PDJointPosControllerConfig(  # good for walk
        #     ["right_knee"],-max_delta,max_delta,damping=base_stiff*80/20,stiffness=base_stiff*80,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_ankle_x = PDJointPosControllerConfig(  # good for walk
        #     ["right_ankle_x"],-max_delta,max_delta,damping=base_stiff*20/20,stiffness=base_stiff*20,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_ankle_y = PDJointPosControllerConfig(  # good for walk
        #     ["right_ankle_y"],-max_delta,max_delta,damping=base_stiff*20/20,stiffness=base_stiff*20,use_delta=True,
        # ))

        # pd_joint_delta_pos.update(left_hip_x = PDJointPosControllerConfig(  # good for walk
        #     ["left_hip_x"],-max_delta,max_delta,damping=base_stiff*40/20,stiffness=base_stiff*40,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_hip_z = PDJointPosControllerConfig(  # good for walk
        #     ["left_hip_z"],-max_delta,max_delta,damping=base_stiff*40/20,stiffness=base_stiff*40,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_hip_y = PDJointPosControllerConfig(  # good for walk
        #     ["left_hip_y"],-max_delta,max_delta,damping=base_stiff*120/20,stiffness=base_stiff*120,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_knee = PDJointPosControllerConfig(  # good for walk
        #     ["left_knee"],-max_delta,max_delta,damping=base_stiff*80/20,stiffness=base_stiff*80,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_ankle_x = PDJointPosControllerConfig(  # good for walk
        #     ["left_ankle_x"],-max_delta,max_delta,damping=base_stiff*20/20,stiffness=base_stiff*20,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_ankle_y = PDJointPosControllerConfig(  # good for walk
        #     ["left_ankle_y"],-max_delta,max_delta,damping=base_stiff*20/20,stiffness=base_stiff*20,use_delta=True,
        # ))

        # pd_joint_delta_pos.update(right_shoulder1 = PDJointPosControllerConfig(  # good for walk
        #     ["right_shoulder1"],-max_delta,max_delta,damping=base_stiff*20/20,stiffness=base_stiff*20,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_shoulder2 = PDJointPosControllerConfig(  # good for walk
        #     ["right_shoulder2"],-max_delta,max_delta,damping=base_stiff*20/20,stiffness=base_stiff*20,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_elbow = PDJointPosControllerConfig(  # good for walk
        #     ["right_elbow"],-max_delta,max_delta,damping=base_stiff*40/20,stiffness=base_stiff*40,use_delta=True,
        # ))

        # pd_joint_delta_pos.update(left_shoulder1 = PDJointPosControllerConfig(  # good for walk
        #     ["left_shoulder1"],-max_delta,max_delta,damping=base_stiff*20/20,stiffness=base_stiff*20,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_shoulder2 = PDJointPosControllerConfig(  # good for walk
        #     ["left_shoulder2"],-max_delta,max_delta,damping=base_stiff*20/20,stiffness=base_stiff*20,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_elbow = PDJointPosControllerConfig(  # good for walk
        #     ["left_elbow"],-max_delta,max_delta,damping=base_stiff*40/20,stiffness=base_stiff*40,use_delta=True,
        # ))

        ###################################################################################################################### good for walk
        # pd_joint_delta_pos.update(abdomen_y = PDJointPosControllerConfig(  # good for walk
        #     ["abdomen_y"],-max_delta,max_delta,damping=5,stiffness=base_stiff*40,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(abdomen_z = PDJointPosControllerConfig(  # good for walk
        #     ["abdomen_z"],-max_delta,max_delta,damping=5,stiffness=base_stiff*40,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(abdomen_x = PDJointPosControllerConfig(  # good for walk
        #     ["abdomen_x"],-max_delta,max_delta,damping=5,stiffness=base_stiff*40,use_delta=True,
        # ))

        # pd_joint_delta_pos.update(right_hip_x = PDJointPosControllerConfig(  # good for walk
        #     ["right_hip_x"],-max_delta,max_delta,damping=5,stiffness=base_stiff*40,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_hip_z = PDJointPosControllerConfig(  # good for walk
        #     ["right_hip_z"],-max_delta,max_delta,damping=5,stiffness=base_stiff*40,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_hip_y = PDJointPosControllerConfig(  # good for walk
        #     ["right_hip_y"],-max_delta,max_delta,damping=5,stiffness=base_stiff*120,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_knee = PDJointPosControllerConfig(  # good for walk
        #     ["right_knee"],-max_delta,max_delta,damping=1,stiffness=base_stiff*80,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_ankle_x = PDJointPosControllerConfig(  # good for walk
        #     ["right_ankle_x"],-max_delta,max_delta,damping=3,stiffness=base_stiff*20,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_ankle_y = PDJointPosControllerConfig(  # good for walk
        #     ["right_ankle_y"],-max_delta,max_delta,damping=3,stiffness=base_stiff*20,use_delta=True,
        # ))

        # pd_joint_delta_pos.update(left_hip_x = PDJointPosControllerConfig(  # good for walk
        #     ["left_hip_x"],-max_delta,max_delta,damping=5,stiffness=base_stiff*40,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_hip_z = PDJointPosControllerConfig(  # good for walk
        #     ["left_hip_z"],-max_delta,max_delta,damping=5,stiffness=base_stiff*40,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_hip_y = PDJointPosControllerConfig(  # good for walk
        #     ["left_hip_y"],-max_delta,max_delta,damping=5,stiffness=base_stiff*120,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_knee = PDJointPosControllerConfig(  # good for walk
        #     ["left_knee"],-max_delta,max_delta,damping=1,stiffness=base_stiff*80,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_ankle_x = PDJointPosControllerConfig(  # good for walk
        #     ["left_ankle_x"],-max_delta,max_delta,damping=3,stiffness=base_stiff*20,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_ankle_y = PDJointPosControllerConfig(  # good for walk
        #     ["left_ankle_y"],-max_delta,max_delta,damping=3,stiffness=base_stiff*20,use_delta=True,
        # ))

        # pd_joint_delta_pos.update(right_shoulder1 = PDJointPosControllerConfig(  # good for walk
        #     ["right_shoulder1"],-max_delta,max_delta,damping=1,stiffness=base_stiff*20,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_shoulder2 = PDJointPosControllerConfig(  # good for walk
        #     ["right_shoulder2"],-max_delta,max_delta,damping=1,stiffness=base_stiff*20,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_elbow = PDJointPosControllerConfig(  # good for walk
        #     ["right_elbow"],-max_delta,max_delta,damping=0,stiffness=base_stiff*40,use_delta=True,
        # ))

        # pd_joint_delta_pos.update(left_shoulder1 = PDJointPosControllerConfig(  # good for walk
        #     ["left_shoulder1"],-max_delta,max_delta,damping=1,stiffness=base_stiff*20,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_shoulder2 = PDJointPosControllerConfig(  # good for walk
        #     ["left_shoulder2"],-max_delta,max_delta,damping=1,stiffness=base_stiff*20,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_elbow = PDJointPosControllerConfig(  # good for walk
        #     ["left_elbow"],-max_delta,max_delta,damping=0,stiffness=base_stiff*40,use_delta=True,
        # ))
        ######################################################################################################################## dampisstiff

        ########################################################################################################################### otherrew_dampisstif_d2
        # base_stiff = 1
        # max_delta = 2
        # pd_joint_delta_pos.update(abdomen_y = PDJointPosControllerConfig(  # good for walk
        #     ["abdomen_y"],-max_delta,max_delta,damping=5,stiffness=base_stiff*40,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(abdomen_z = PDJointPosControllerConfig(  # good for walk
        #     ["abdomen_z"],-max_delta,max_delta,damping=5,stiffness=base_stiff*40,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(abdomen_x = PDJointPosControllerConfig(  # good for walk
        #     ["abdomen_x"],-max_delta,max_delta,damping=5,stiffness=base_stiff*40,use_delta=True,
        # ))

        # pd_joint_delta_pos.update(right_hip_x = PDJointPosControllerConfig(  # good for walk
        #     ["right_hip_x"],-max_delta,max_delta,damping=5,stiffness=base_stiff*40,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_hip_z = PDJointPosControllerConfig(  # good for walk
        #     ["right_hip_z"],-max_delta,max_delta,damping=5,stiffness=base_stiff*40,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_hip_y = PDJointPosControllerConfig(  # good for walk
        #     ["right_hip_y"],-max_delta,max_delta,damping=5,stiffness=base_stiff*120,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_knee = PDJointPosControllerConfig(  # good for walk
        #     ["right_knee"],-max_delta,max_delta,damping=1,stiffness=base_stiff*80,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_ankle_x = PDJointPosControllerConfig(  # good for walk
        #     ["right_ankle_x"],-max_delta,max_delta,damping=3,stiffness=base_stiff*20,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_ankle_y = PDJointPosControllerConfig(  # good for walk
        #     ["right_ankle_y"],-max_delta,max_delta,damping=3,stiffness=base_stiff*20,use_delta=True,
        # ))

        # pd_joint_delta_pos.update(left_hip_x = PDJointPosControllerConfig(  # good for walk
        #     ["left_hip_x"],-max_delta,max_delta,damping=5,stiffness=base_stiff*40,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_hip_z = PDJointPosControllerConfig(  # good for walk
        #     ["left_hip_z"],-max_delta,max_delta,damping=5,stiffness=base_stiff*40,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_hip_y = PDJointPosControllerConfig(  # good for walk
        #     ["left_hip_y"],-max_delta,max_delta,damping=5,stiffness=base_stiff*120,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_knee = PDJointPosControllerConfig(  # good for walk
        #     ["left_knee"],-max_delta,max_delta,damping=1,stiffness=base_stiff*80,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_ankle_x = PDJointPosControllerConfig(  # good for walk
        #     ["left_ankle_x"],-max_delta,max_delta,damping=3,stiffness=base_stiff*20,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_ankle_y = PDJointPosControllerConfig(  # good for walk
        #     ["left_ankle_y"],-max_delta,max_delta,damping=3,stiffness=base_stiff*20,use_delta=True,
        # ))

        # pd_joint_delta_pos.update(right_shoulder1 = PDJointPosControllerConfig(  # good for walk
        #     ["right_shoulder1"],-max_delta,max_delta,damping=1,stiffness=base_stiff*20,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_shoulder2 = PDJointPosControllerConfig(  # good for walk
        #     ["right_shoulder2"],-max_delta,max_delta,damping=1,stiffness=base_stiff*20,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(right_elbow = PDJointPosControllerConfig(  # good for walk
        #     ["right_elbow"],-max_delta,max_delta,damping=0,stiffness=base_stiff*40,use_delta=True,
        # ))

        # pd_joint_delta_pos.update(left_shoulder1 = PDJointPosControllerConfig(  # good for walk
        #     ["left_shoulder1"],-max_delta,max_delta,damping=1,stiffness=base_stiff*20,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_shoulder2 = PDJointPosControllerConfig(  # good for walk
        #     ["left_shoulder2"],-max_delta,max_delta,damping=1,stiffness=base_stiff*20,use_delta=True,
        # ))
        # pd_joint_delta_pos.update(left_elbow = PDJointPosControllerConfig(  # good for walk
        #     ["left_elbow"],-max_delta,max_delta,damping=0,stiffness=base_stiff*40,use_delta=True,
        # ))
        ########################################################################################################################### otherrew_dampisstif_d2

        base_stiff = 1
        max_delta = 2
        pd_joint_delta_pos.update(
            abdomen_y=PDJointPosControllerConfig(  # good for walk
                ["abdomen_y"],
                -max_delta,
                max_delta,
                damping=5,
                stiffness=base_stiff * 40,
                use_delta=True,
            )
        )
        pd_joint_delta_pos.update(
            abdomen_z=PDJointPosControllerConfig(  # good for walk
                ["abdomen_z"],
                -max_delta,
                max_delta,
                damping=5,
                stiffness=base_stiff * 40,
                use_delta=True,
            )
        )
        pd_joint_delta_pos.update(
            abdomen_x=PDJointPosControllerConfig(  # good for walk
                ["abdomen_x"],
                -max_delta,
                max_delta,
                damping=5,
                stiffness=base_stiff * 40,
                use_delta=True,
            )
        )

        pd_joint_delta_pos.update(
            right_hip_x=PDJointPosControllerConfig(  # good for walk
                ["right_hip_x"],
                -max_delta,
                max_delta,
                damping=5,
                stiffness=base_stiff * 40,
                use_delta=True,
            )
        )
        pd_joint_delta_pos.update(
            right_hip_z=PDJointPosControllerConfig(  # good for walk
                ["right_hip_z"],
                -max_delta,
                max_delta,
                damping=5,
                stiffness=base_stiff * 40,
                use_delta=True,
            )
        )
        pd_joint_delta_pos.update(
            right_hip_y=PDJointPosControllerConfig(  # good for walk
                ["right_hip_y"],
                -max_delta,
                max_delta,
                damping=5,
                stiffness=base_stiff * 120,
                use_delta=True,
            )
        )
        pd_joint_delta_pos.update(
            right_knee=PDJointPosControllerConfig(  # good for walk
                ["right_knee"],
                -max_delta,
                max_delta,
                damping=1,
                stiffness=base_stiff * 80,
                use_delta=True,
            )
        )
        pd_joint_delta_pos.update(
            right_ankle_x=PDJointPosControllerConfig(  # good for walk
                ["right_ankle_x"],
                -max_delta,
                max_delta,
                damping=3,
                stiffness=base_stiff * 20,
                use_delta=True,
            )
        )
        pd_joint_delta_pos.update(
            right_ankle_y=PDJointPosControllerConfig(  # good for walk
                ["right_ankle_y"],
                -max_delta,
                max_delta,
                damping=3,
                stiffness=base_stiff * 20,
                use_delta=True,
            )
        )

        pd_joint_delta_pos.update(
            left_hip_x=PDJointPosControllerConfig(  # good for walk
                ["left_hip_x"],
                -max_delta,
                max_delta,
                damping=5,
                stiffness=base_stiff * 40,
                use_delta=True,
            )
        )
        pd_joint_delta_pos.update(
            left_hip_z=PDJointPosControllerConfig(  # good for walk
                ["left_hip_z"],
                -max_delta,
                max_delta,
                damping=5,
                stiffness=base_stiff * 40,
                use_delta=True,
            )
        )
        pd_joint_delta_pos.update(
            left_hip_y=PDJointPosControllerConfig(  # good for walk
                ["left_hip_y"],
                -max_delta,
                max_delta,
                damping=5,
                stiffness=base_stiff * 120,
                use_delta=True,
            )
        )
        pd_joint_delta_pos.update(
            left_knee=PDJointPosControllerConfig(  # good for walk
                ["left_knee"],
                -max_delta,
                max_delta,
                damping=1,
                stiffness=base_stiff * 80,
                use_delta=True,
            )
        )
        pd_joint_delta_pos.update(
            left_ankle_x=PDJointPosControllerConfig(  # good for walk
                ["left_ankle_x"],
                -max_delta,
                max_delta,
                damping=3,
                stiffness=base_stiff * 20,
                use_delta=True,
            )
        )
        pd_joint_delta_pos.update(
            left_ankle_y=PDJointPosControllerConfig(  # good for walk
                ["left_ankle_y"],
                -max_delta,
                max_delta,
                damping=3,
                stiffness=base_stiff * 20,
                use_delta=True,
            )
        )

        pd_joint_delta_pos.update(
            right_shoulder1=PDJointPosControllerConfig(  # good for walk
                ["right_shoulder1"],
                -max_delta,
                max_delta,
                damping=1,
                stiffness=base_stiff * 20,
                use_delta=True,
            )
        )
        pd_joint_delta_pos.update(
            right_shoulder2=PDJointPosControllerConfig(  # good for walk
                ["right_shoulder2"],
                -max_delta,
                max_delta,
                damping=1,
                stiffness=base_stiff * 20,
                use_delta=True,
            )
        )
        pd_joint_delta_pos.update(
            right_elbow=PDJointPosControllerConfig(  # good for walk
                ["right_elbow"],
                -max_delta,
                max_delta,
                damping=0,
                stiffness=base_stiff * 40,
                use_delta=True,
            )
        )

        pd_joint_delta_pos.update(
            left_shoulder1=PDJointPosControllerConfig(  # good for walk
                ["left_shoulder1"],
                -max_delta,
                max_delta,
                damping=1,
                stiffness=base_stiff * 20,
                use_delta=True,
            )
        )
        pd_joint_delta_pos.update(
            left_shoulder2=PDJointPosControllerConfig(  # good for walk
                ["left_shoulder2"],
                -max_delta,
                max_delta,
                damping=1,
                stiffness=base_stiff * 20,
                use_delta=True,
            )
        )
        pd_joint_delta_pos.update(
            left_elbow=PDJointPosControllerConfig(  # good for walk
                ["left_elbow"],
                -max_delta,
                max_delta,
                damping=0,
                stiffness=base_stiff * 40,
                use_delta=True,
            )
        )

        pd_joint_delta_pos.update(balance_passive_force=False)

        return deepcopy_dict(
            dict(
                pd_joint_pos=dict(body=pd_joint_pos, balance_passive_force=False),
                pd_joint_delta_pos=pd_joint_delta_pos,
            )
        )
