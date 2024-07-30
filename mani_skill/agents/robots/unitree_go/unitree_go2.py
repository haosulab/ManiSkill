import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


@register_agent(asset_download_ids=["unitree_go2"])
class UnitreeGo2(BaseAgent):
    uid = "unitree_go2"
    urdf_path = f"{ASSET_DIR}/robots/unitree_go2/urdf/go2_description.urdf"
    urdf_config = dict(
        _materials=dict(
            foot=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            FR_foot=dict(material="foot", patch_radius=0.1, min_patch_radius=0.1),
            RR_foot=dict(material="foot", patch_radius=0.1, min_patch_radius=0.1),
            RL_foot=dict(material="foot", patch_radius=0.1, min_patch_radius=0.1),
            FL_foot=dict(material="foot", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )

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
            pd_joint_delta_pos=dict(
                body=PDJointPosControllerConfig(
                    [x.name for x in self.robot.active_joints],
                    lower=-0.7,
                    upper=0.7,
                    stiffness=1000,
                    damping=100,
                    normalize_action=True,
                    use_delta=True,
                ),
                balance_passive_force=False,
            ),
            pd_joint_pos=dict(
                body=PDJointPosControllerConfig(
                    [x.name for x in self.robot.active_joints],
                    lower=None,
                    upper=None,
                    stiffness=1000,
                    damping=100,
                    normalize_action=False,
                ),
                balance_passive_force=False,
            ),
        )

    def is_fallen(self):
        """This quadruped is considered fallen if its body contacts the ground"""
        forces = self.robot.get_net_contact_forces(["base"])
        return torch.norm(forces, dim=-1).max(-1).values > 1


@register_agent(asset_download_ids=["unitree_go2"])
class UnitreeGo2Simplified(UnitreeGo2):
    """
    The UnitreeGo2 robot with heavily simplified collision meshes to enable faster simulation and easier locomotion.
    """

    uid = "unitree_go2_simplified_locomotion"
    urdf_path = f"{ASSET_DIR}/robots/unitree_go2/urdf/go2_description_simplified_locomotion.urdf"
