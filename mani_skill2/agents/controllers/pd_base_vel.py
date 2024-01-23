import numpy as np
import torch

from mani_skill2.utils.geometry import rotate_2d_vec_by_angle

from .pd_joint_vel import PDJointVelController, PDJointVelControllerConfig

# TODO (stao): add GPU support here


class PDBaseVelController(PDJointVelController):
    """PDJointVelController for ego-centric base movement."""

    def _initialize_action_space(self):
        # At least support xy-plane translation and z-axis rotation
        assert len(self.joints) >= 3, len(self.joints)
        super()._initialize_action_space()

    def set_action(self, action: np.ndarray):
        action = self._preprocess_action(action)

        # TODO (arth): add support for batched qpos and gpu sim
        if isinstance(self.qpos, torch.Tensor):
            qpos = self.qpos.detach().cpu().numpy()
        qpos = qpos[0]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        # Convert to ego-centric action
        # Assume the 3rd DoF stands for orientation
        ori = qpos[2]
        vel = rotate_2d_vec_by_angle(action[:2], ori)
        new_action = np.hstack([vel, action[2:]])

        for i, joint in enumerate(self.joints):
            joint.set_drive_velocity_target(np.array([new_action[i]]))


class PDBaseVelControllerConfig(PDJointVelControllerConfig):
    controller_cls = PDBaseVelController
