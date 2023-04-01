from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
from gymnasium import spaces

from ..base_controller import BaseController, ControllerConfig
from .pd_joint_pos import PDJointPosController, PDJointPosControllerConfig


class PDJointPosVelController(PDJointPosController):
    config: "PDJointPosVelControllerConfig"

    def _initialize_action_space(self):
        joint_limits = self._get_joint_limits()
        pos_low, pos_high = joint_limits[:, 0], joint_limits[:, 1]
        vel_low = np.broadcast_to(self.config.vel_lower, pos_low.shape)
        vel_high = np.broadcast_to(self.config.vel_upper, pos_high.shape)
        low = np.float32(np.hstack([pos_low, vel_low]))
        high = np.float32(np.hstack([pos_high, vel_high]))
        self.action_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self):
        super().reset()
        self._target_qvel = np.zeros_like(self._target_qpos)

    def set_drive_velocity_targets(self, targets):
        for i, joint in enumerate(self.joints):
            joint.set_drive_velocity_target(targets[i])

    def set_action(self, action: np.ndarray):
        action = self._preprocess_action(action)
        nq = len(action) // 2

        self._step = 0
        self._start_qpos = self.qpos

        if self.config.use_delta:
            if self.config.use_target:
                self._target_qpos = self._target_qpos + action[:nq]
            else:
                self._target_qpos = self._start_qpos + action[:nq]
        else:
            # Compatible with mimic
            self._target_qpos = np.broadcast_to(action[:nq], self._start_qpos.shape)

        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

        self._target_qvel = action[nq:]
        self.set_drive_velocity_targets(self._target_qvel)


@dataclass
class PDJointPosVelControllerConfig(PDJointPosControllerConfig):
    controller_cls = PDJointPosVelController
    vel_lower: Union[float, Sequence[float]] = -1.0
    vel_upper: Union[float, Sequence[float]] = 1.0
