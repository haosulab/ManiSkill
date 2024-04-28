import numpy as np
import sapien

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


@register_agent()  # uncomment this if you want to register the agent so you can instantiate it by ID when creating environments
class UnitreeGo2(BaseAgent):
    uid = "unitree_go2"
    urdf_path = f"{ASSET_DIR}/robots/unitree_go2/urdf/go2_description.urdf"  # You can use f"{PACKAGE_ASSET_DIR}" to reference a urdf file in the mani_skill /assets package folder

    # you may need to use this modify the friction values of some links in order to make it possible to e.g. grasp objects or avoid sliding on the floor
    urdf_config = dict()

    fix_root_link = False

    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.29]),
            qpos=np.array(
                [0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, -1.8, -1.8, -1.8, -1.8]
            ),
        )
    )

    @property
    def _controller_configs(
        self,
    ):

        return dict(
            pd_joint_pos=dict(
                body=PDJointPosControllerConfig(
                    [x.name for x in self.robot.active_joints],
                    lower=None,
                    upper=None,
                    stiffness=100,
                    damping=10,
                    normalize_action=False,
                ),
                balance_passive_force=False,
            ),
            pd_joint_delta_pos=dict(
                body=PDJointPosControllerConfig(
                    [x.name for x in self.robot.active_joints],
                    lower=-0.1,
                    upper=0.1,
                    stiffness=20,
                    damping=5,
                    normalize_action=True,
                    use_delta=True,
                ),
                balance_passive_force=False,
            ),
        )
