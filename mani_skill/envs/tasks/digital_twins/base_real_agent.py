from typing import Optional, Union

import cv2
import numpy as np
import torch

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.scene import ManiSkillScene


# converting sim controller to real controller, only requires changing a few functions
class RealController:
    def __init__(self, controller, real_agent):
        self.controller_instance = controller
        self.real_agent = real_agent

    @property
    def qpos(self):
        return self.real_agent.qpos

    @property
    def qvel(self):
        return self.real_agent.qvel

    @property
    def device(self):
        return torch.device("cpu")

    def __getattr__(self, name):
        return getattr(self.controller_instance, name)


class BaseRealAgent:
    def __init__(
        self,
        sim_agent_cls: BaseAgent,
        control_freq: Optional[int] = None,
        control_mode: Optional[str] = None,
    ):
        self.sim_agent = sim_agent_cls(
            ManiSkillScene(), control_freq, control_mode, None, None
        )
        self.robot = self._load_agent()

        # reuse of ms3 controller for real robot
        self.control_mode = (
            control_mode if control_mode is not None else self.sim_agent.control_mode
        )
        self.sim_controller = self.sim_agent.controllers[self.control_mode]
        self.controller = RealController(self.sim_controller, self)
        self.control_freq = control_freq

    @property
    def target_qpos(self):
        return self.controller._target_qpos.clone()

    def _load_agent(self, yaml_path, **kwargs):
        raise NotImplementedError()

    def send_qpos(qpos):
        """send qpos for robot to match"""
        raise NotImplementedError()

    @property
    def qpos(self):
        """Read current qpos of robot"""
        raise NotImplementedError()

    @property
    def qvel(self):
        """Read current qvel of robot"""
        raise NotImplementedError()

    def reset(self, qpos: Union[torch.Tensor, np.ndarray] = None):
        raise NotImplementedError()

    def render(self):
        raise NotImplementedError()

    def get_obs_sensor_data(self):
        raise NotImplementedError()
