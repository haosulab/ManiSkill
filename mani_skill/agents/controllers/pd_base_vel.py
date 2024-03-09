import numpy as np
import torch

from mani_skill.utils.geometry import rotate_2d_vec_by_angle
from mani_skill.utils.structs.types import Array

from .pd_joint_vel import PDJointVelController, PDJointVelControllerConfig


class PDBaseVelController(PDJointVelController):
    """PDJointVelController for ego-centric base movement."""

    def _initialize_action_space(self):
        # At least support xy-plane translation and z-axis rotation
        assert len(self.joints) >= 3, len(self.joints)
        super()._initialize_action_space()

    def set_action(self, action: Array):
        action = self._preprocess_action(action)
        # Convert to ego-centric action
        # Assume the 3rd DoF stands for orientation
        ori = self.qpos[:, 2]
        rot_mat = torch.zeros(ori.shape[0], 2, 2, device=action.device)
        rot_mat[:, 0, 0] = torch.cos(ori)
        rot_mat[:, 0, 1] = -torch.sin(ori)
        rot_mat[:, 1, 0] = torch.sin(ori)
        rot_mat[:, 1, 1] = torch.cos(ori)
        vel = (rot_mat @ action[:, :2].float().unsqueeze(-1)).squeeze(-1)
        new_action = torch.hstack([vel, action[:, 2:]])
        self.articulation.set_joint_drive_velocity_targets(
            new_action, self.joints, self.joint_indices
        )


class PDBaseVelControllerConfig(PDJointVelControllerConfig):
    controller_cls = PDBaseVelController
