import numpy as np
import torch

from gymnasium import spaces
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
            new_action, self.joints, self.active_joint_indices
        )


class PDBaseVelControllerConfig(PDJointVelControllerConfig):
    controller_cls = PDBaseVelController


class PDBaseForwardVelController(PDJointVelController):
    """PDJointVelController for forward-only ego-centric base movement."""

    def _initialize_action_space(self):
        assert len(self.joints) >= 3, len(self.joints)
        low = np.float32(np.broadcast_to(self.config.lower, 2))
        high = np.float32(np.broadcast_to(self.config.upper, 2))
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def set_action(self, action: Array):
        action = self._preprocess_action(action)
        # action[:, 0] should correspond to forward vel
        # action[:, 1] should correspond to rotation vel

        # Convert to ego-centric action
        # Assume the 3rd DoF stands for orientation
        ori = self.qpos[:, 2]
        rot_mat = torch.zeros(ori.shape[0], 2, 2, device=action.device)
        rot_mat[:, 0, 0] = torch.cos(ori)
        rot_mat[:, 0, 1] = -torch.sin(ori)
        rot_mat[:, 1, 0] = torch.sin(ori)
        rot_mat[:, 1, 1] = torch.cos(ori)

        # Assume the 1st DoF stands for forward movement
        # make action with 0 y vel
        move_action = action.clone()
        move_action[:, 1] = 0
        vel = (rot_mat @ move_action.float().unsqueeze(-1)).squeeze(-1)
        new_action = torch.hstack([vel, action[:, 1:]])
        self.articulation.set_joint_drive_velocity_targets(
            new_action, self.joints, self.active_joint_indices
        )


class PDBaseForwardVelControllerConfig(PDJointVelControllerConfig):
    controller_cls = PDBaseForwardVelController
