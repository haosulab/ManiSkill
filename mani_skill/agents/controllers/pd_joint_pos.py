from dataclasses import dataclass, field
from typing import Dict, Sequence, Union

import numpy as np
import torch
from gymnasium import spaces

from mani_skill.utils import common
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.structs.types import Array, DriveMode

from .base_controller import BaseController, ControllerConfig


class PDJointPosController(BaseController):
    config: "PDJointPosControllerConfig"
    _start_qpos = None
    _target_qpos = None

    def _get_joint_limits(self):
        qlimits = (
            self.articulation.get_qlimits()[0, self.active_joint_indices].cpu().numpy()
        )
        # Override if specified
        if self.config.lower is not None:
            qlimits[:, 0] = self.config.lower
        if self.config.upper is not None:
            qlimits[:, 1] = self.config.upper
        return qlimits

    def _initialize_action_space(self):
        joint_limits = self._get_joint_limits()
        low, high = joint_limits[:, 0], joint_limits[:, 1]
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def set_drive_property(self):
        n = len(self.joints)
        stiffness = np.broadcast_to(self.config.stiffness, n)
        damping = np.broadcast_to(self.config.damping, n)
        force_limit = np.broadcast_to(self.config.force_limit, n)
        friction = np.broadcast_to(self.config.friction, n)

        for i, joint in enumerate(self.joints):
            drive_mode = self.config.drive_mode
            if not isinstance(drive_mode, str):
                drive_mode = drive_mode[i]
            joint.set_drive_properties(
                stiffness[i], damping[i], force_limit=force_limit[i], mode=drive_mode
            )
            joint.set_friction(friction[i])

    def reset(self):
        super().reset()
        self._step = 0  # counter of simulation steps after action is set
        if self._start_qpos is None:
            self._start_qpos = self.qpos.clone()
        else:

            self._start_qpos[self.scene._reset_mask] = self.qpos[
                self.scene._reset_mask
            ].clone()
        if self._target_qpos is None:
            self._target_qpos = self.qpos.clone()
        else:
            self._target_qpos[self.scene._reset_mask] = self.qpos[
                self.scene._reset_mask
            ].clone()

    def set_drive_targets(self, targets):
        self.articulation.set_joint_drive_targets(
            targets, self.joints, self.active_joint_indices
        )

    def set_action(self, action: Array):
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos
        if self.config.use_delta:
            if self.config.use_target:
                self._target_qpos = self._target_qpos + action
            else:
                self._target_qpos = self._start_qpos + action
        else:
            # Compatible with mimic controllers. Need to clone here otherwise cannot do in-place replacements in the reset function
            self._target_qpos = torch.broadcast_to(
                action, self._start_qpos.shape
            ).clone()
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

    def before_simulation_step(self):
        self._step += 1

        # Compute the next target via a linear interpolation
        if self.config.interpolate:
            targets = self._start_qpos + self._step_size * self._step
            self.set_drive_targets(targets)

    def get_state(self) -> dict:
        if self.config.use_target:
            return {"target_qpos": self._target_qpos}
        return {}

    def set_state(self, state: dict):
        if self.config.use_target:
            self._target_qpos = state["target_qpos"]


@dataclass
class PDJointPosControllerConfig(ControllerConfig):
    lower: Union[None, float, Sequence[float]]
    upper: Union[None, float, Sequence[float]]
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    use_delta: bool = False
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = True
    drive_mode: Union[Sequence[DriveMode], DriveMode] = "force"
    controller_cls = PDJointPosController


class PDJointPosMimicController(PDJointPosController):
    config: "PDJointPosMimicControllerConfig"

    def _initialize_joints(self):
        super()._initialize_joints()
        # do some sanity checks and setup the mimic controller

        self.mimic_joint_indices = []
        self.mimic_control_joint_indices = []
        if len(self.config.mimic_targets) == 0:
            if len(self.config.joint_names) == 2:
                logger.warning(
                    f"Mimic targets dictionary is missing for controller config for {self.articulation.name}. Assuming the first joint is the control joint and the second joint is the mimic joint"
                )
                self.config.mimic_targets = {
                    self.config.joint_names[0]: self.config.joint_names[1]
                }
            else:
                raise ValueError(
                    "Mimic targets dictionary is missing. Please provide a mimic targets dictionary to setup mimic controllers withh more than 2 joints"
                )

        self._multiplier = torch.ones(
            len(self.config.mimic_targets), device=self.device
        )
        self._offset = torch.zeros(len(self.config.mimic_targets), device=self.device)

        for mimic_joint_name, control_joint_name in self.config.mimic_targets.items():
            assert (
                self.articulation.joints_map[mimic_joint_name].active_index is not None
            ), f"Mimic joint {mimic_joint_name} is not active, cannot be used as a mimic joint in a mimic controller"
            assert (
                self.articulation.joints_map[control_joint_name].active_index
                is not None
            ), f"Control joint {control_joint_name} is not active, cannot be used as a control joint in a mimic controller"

            # NOTE (stao): we are assuming all the articulations are the exact same structure. At the moment I see very little reason to try and support training/evaluating on different embodiments
            # simultaneously in one process
            self.mimic_joint_indices.append(
                torch.argwhere(
                    self.active_joint_indices
                    == self.articulation.joints_map[mimic_joint_name].active_index[0]
                )
            )
            self.mimic_control_joint_indices.append(
                torch.argwhere(
                    self.active_joint_indices
                    == self.articulation.joints_map[control_joint_name].active_index[0]
                )
            )

            if isinstance(self.config.multiplier, dict):
                self._multiplier[self.mimic_joint_indices[-1]] = self.config.multiplier[
                    mimic_joint_name
                ]
            if isinstance(self.config.offset, dict):
                self._offset[self.mimic_joint_indices[-1]] = self.config.offset[
                    mimic_joint_name
                ]

        self.mimic_joint_indices = torch.tensor(
            self.mimic_joint_indices, dtype=torch.int32, device=self.device
        )
        self.mimic_control_joint_indices = torch.tensor(
            self.mimic_control_joint_indices, dtype=torch.int32, device=self.device
        )
        """list of control joint indices corresponding to mimic joint indices"""
        self.control_joint_indices = torch.unique(self.mimic_control_joint_indices).to(
            torch.int32
        )
        """list of all directly controlled joint indices"""

        if isinstance(self.config.multiplier, float):
            self._multiplier[:] = self.config.multiplier
        if isinstance(self.config.offset, float):
            self._offset[:] = self.config.offset

    def _get_joint_limits(self):
        joint_limits = super()._get_joint_limits()
        return joint_limits[0:1]

    def set_action(self, action: Array):
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos
        if self.config.use_delta:
            if self.config.use_target:
                self._target_qpos[:, self.control_joint_indices] += action
                self._target_qpos[:, self.mimic_joint_indices] = (
                    self._target_qpos[:, self.mimic_control_joint_indices]
                    * self.config.multiplier[None, :]
                    + self.config.offset[None, :]
                )
            else:
                self._target_qpos[:, self.control_joint_indices] = (
                    self._start_qpos[:, self.control_joint_indices] + action
                )
        else:
            self._target_qpos[:, self.control_joint_indices] = action
            self._target_qpos[:, self.mimic_joint_indices] = (
                self._target_qpos[:, self.mimic_control_joint_indices]
                * self._multiplier[None, :]
                + self._offset[None, :]
            )
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)


@dataclass
class PDJointPosMimicControllerConfig(PDJointPosControllerConfig):
    """
    PD Joint Position mimic controller configuration. This kind of controller is used to emuluate a mimic joint. For some simulation backends mimic joints are not
    well simulated and/or are hard to tune and so a mimic controller can be used for better stability as well as alignment with a real robot.

    The `joint_names` field is expected to be a list of all the joints, whether they are the joints to be controlled or the joints that are controlled via mimicing.

    `mimic_targets` is a dictionary that maps the mimic joint name to the controlling joint name and is required. All given joints in
    `joint_names` must be in the `mimic_targets` dictionary as either a key or a value. The ones that are keys are referred to as the control joints and the ones that are values
    are referred to as the mimic joints. Control joints are the ones directly controlled by the agent/user and mimic joints are implicitly controlled by following the control joints via the following equation:

    q_mimic = q_controlling * multiplier + offset


    To set multiplier you can either provide a single float to set a global multiplier or a dictionary to set a per-mimic-joint multiplier, which maps the mimic joint name to its multiplier.

    To set offset you can either provide a single float to set a global offset or a dictionary to set a per-mimic-joint offset, which maps the mimic joint name to its offset.
    """

    controller_cls = PDJointPosMimicController
    mimic_targets: Dict[str, str] = field(default_factory=dict)
    """the mimic targets. Maps the actual mimic joint name to the controlling joint name"""
    multiplier: Union[float, Dict[str, float]] = 1.0
    """the multiplier for the mimic control. Defaults to 1.0. Can also be a dictionary to define per-mimic-joint multipliers"""
    offset: Union[float, Dict[str, float]] = 0.0
    """the offset for the mimic control. Defaults to 0.0. Can also be a dictionary to define per-mimic-joint offsets"""
