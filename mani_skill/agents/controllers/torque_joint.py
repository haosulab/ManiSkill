from dataclasses import dataclass
from typing import Sequence, Union
from copy import deepcopy

import numpy as np
import torch
from gymnasium import spaces

from mani_skill.utils import common
from mani_skill.utils.structs.types import Array, DriveMode

from mani_skill.agents.controllers.base_controller import BaseController, ControllerConfig, DictController, CombinedController

from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda import Panda

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs import Articulation
import sapien.physx as physx

class TorqueJointController(BaseController):
    config: "TorqueJointControllerConfig"
    _target_qf = None
    _qf_error = None
    _start_qf = None
    _full_qf = None
    def __init__(
        self,
        config: "ControllerConfig",
        articulation: Articulation,
        control_freq: int,
        sim_freq: int = None,
        scene: ManiSkillScene = None,
    ):
        super().__init__(config, articulation, control_freq, sim_freq, scene)

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
        """ 
            Creates an action space for Gym env obj
            must be a spaces.box
        """
        joint_limits = self._get_joint_limits()
        low, high = joint_limits[:, 0], joint_limits[:, 1]
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def set_drive_property(self):
        """
            Set drive properties
            - max_saturation (max change per step in torque)
            - torque limit
            Then sets each joints properties
        """
        n = len(self.joints)
        #self.force_limit = np.broadcast_to(self.config.force_limit, n)
        self.max_sat = torch.ones(n, 
                dtype=torch.float32,
                device=self.scene.device
            ) * self.config.max_saturation

    def reset(self):
        """
            Resets the controller to an initial state. 
            This is called upon environment creation 
            and each environment reset
        """
        super().reset()
        self._step=0
        if self._full_qf is None:
            self._full_qf = self.articulation.get_qf()

        if self._start_qf is None:
            self._start_qf = self.qf.clone()
        else:
            self._start_qf[self.scene._reset_mask] = self.qf[
                    self.scene._reset_mask
            ].clone()
        if self._target_qf is None:
            self._target_qf = self.qf.clone()
        else:
            self._target_qf[self.scene._reset_mask] = self.qf[
                self.scene._reset_mask
            ].clone()

    def set_action(self, action: Array):
        """
            Convert action to tensor
            Any preprocessing of action 
            befor setting the drive targets
        """
        action = self._preprocess_action(action)
        action = common.to_tensor(action)
        self._start_qf = self.qf
        
        if self.config.use_delta:
            if self.config.use_target:
                self._target_qf = self._target_qf + action
            else:
                self._target_qf = self._start_qf + action
        else:
            # Compatible with mimic controllers. Need to clone here 
            # otherwise cannot do in-place replacements in the reset 
            # function
            self._target_qf = torch.broadcast_to(
                action, self._start_qf.shape
            ).clone() 

    def get_state(self) -> dict:
        """ Returns the targets """
        if self.config.use_target:
            return {"target_qf": self._target_qf}
        return {}

    def set_state(self, state: dict):
        """ Sets the targets """
        if self.config.use_target:
            self._target_qf = state["target_qf"]

    def set_drive_targets(self, targets):
        """ Set target for each joint """
        self._target_qf = targets

    @property
    def qf(self):
        return self.articulation.get_qf()[:,self.active_joint_indices]
    
    def before_simulation_step(self):
        self._qf_error = self._target_qf - self.qf

        # clip error
        self._qf_error = torch.clamp(self._qf_error, 
                                    -self.max_sat,
                                    self.max_sat
        )
        # set new qf
        self._full_qf[:,self.active_joint_indices] = self.qf + self._qf_error
        self.articulation.set_qf(self._full_qf)

        if type(self.scene.px) == physx.PhysxGpuSystem:
            self.scene.px.gpu_apply_articulation_qf()


@dataclass
class TorqueJointControllerConfig(ControllerConfig):
    lower: Union[None, float, Sequence[float]]
    upper: Union[None, float, Sequence[float]]
    max_saturation: float = 10.0
    use_delta: bool = False
    use_target: bool = False
    #interpolate: bool = False
    normalize_action: bool = True
    drive_mode: Union[Sequence[DriveMode], DriveMode] = "force"
    controller_cls = TorqueJointController