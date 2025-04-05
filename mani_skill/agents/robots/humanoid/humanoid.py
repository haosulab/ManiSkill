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
            pose=sapien.Pose(p=[0, 0, 1.13]),
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

        # for pd_joint_delta_pos control
        joints_dict = {
            "abdomen_y": {"damping": 5, "stiffness": 40},
            "abdomen_z": {"damping": 5, "stiffness": 40},
            "abdomen_x": {"damping": 5, "stiffness": 40},
            "right_hip_x": {"damping": 5, "stiffness": 40},
            "right_hip_z": {"damping": 5, "stiffness": 40},
            "right_hip_y": {"damping": 5, "stiffness": 120},
            "right_knee": {"damping": 1, "stiffness": 80},
            "right_ankle_x": {"damping": 3, "stiffness": 20},
            "right_ankle_y": {"damping": 3, "stiffness": 40},
            "left_hip_x": {"damping": 5, "stiffness": 40},
            "left_hip_z": {"damping": 5, "stiffness": 40},
            "left_hip_y": {"damping": 5, "stiffness": 120},
            "left_knee": {"damping": 1, "stiffness": 80},
            "left_ankle_x": {"damping": 3, "stiffness": 20},
            "left_ankle_y": {"damping": 3, "stiffness": 40},
            "right_shoulder1": {"damping": 1, "stiffness": 20},
            "right_shoulder2": {"damping": 1, "stiffness": 20},
            "right_elbow": {"damping": 0, "stiffness": 40},
            "left_shoulder1": {"damping": 1, "stiffness": 20},
            "left_shoulder2": {"damping": 1, "stiffness": 20},
            "left_elbow": {"damping": 0, "stiffness": 40},
        }

        joint_names = list(joints_dict.keys())
        assert sorted(joint_names) == sorted([x.name for x in self.robot.active_joints])

        damping = np.array([joint["damping"] for joint in joints_dict.values()])
        stiffness = np.array([joint["stiffness"] for joint in joints_dict.values()])

        pd_joint_delta_pos = PDJointPosControllerConfig(
            joint_names,
            -2,
            2,
            damping=damping,
            stiffness=stiffness,
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
