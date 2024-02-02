from collections import OrderedDict
from typing import Dict, Generic, List, TypeVar

import torch
from gymnasium import spaces

from mani_skill2.agents.base_agent import BaseAgent

T = TypeVar("T")


class MultiAgent(BaseAgent, Generic[T]):
    agents: T

    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
        self.agents_dict: Dict[str, BaseAgent] = OrderedDict()
        self.scene = agents[0].scene
        self.sensor_configs = []
        for i, agent in enumerate(self.agents):
            self.sensor_configs += agent.sensor_configs
            self.agents_dict[f"{agent.uid}-{i}"] = agent

    def get_proprioception(self):
        proprioception = OrderedDict()
        for i, agent in enumerate(self.agents):
            proprioception[f"{agent.uid}-{i}"] = agent.get_proprioception()
        return proprioception

    def initialize(self):
        for agent in self.agents:
            agent.initialize()

    @property
    def control_mode(self):
        """Get the currently activated controller uid of each robot"""
        return {uid: agent.control_mode for uid, agent in self.agents_dict.items()}

    def set_control_mode(self, control_mode: List[str]):
        assert len(control_mode) == len(
            self.agents
        ), "For task with multiple agents, setting control mode on the MultiAgent object requires a control mode for each agent"
        for cm, agent in zip(control_mode, self.agents):
            agent.set_control_mode(cm)

    @property
    def controller(self):
        """Get currently activated controller."""
        if self._control_mode is None:
            raise RuntimeError("Please specify a control mode first")
        else:
            return self.controllers[self._control_mode]

    @property
    def action_space(self):
        return spaces.Dict(
            {uid: agent.action_space for uid, agent in self.agents_dict.items()}
        )

    @property
    def single_action_space(self):
        return spaces.Dict(
            {uid: agent.single_action_space for uid, agent in self.agents_dict.items()}
        )

    def set_action(self, action):
        """
        Set the agent's action which is to be executed in the next environment timestep
        """
        for uid, agent in self.agents_dict.items():
            agent.set_action(action[uid])

    def before_simulation_step(self):
        for agent in self.agents:
            agent.controller.before_simulation_step()

    # -------------------------------------------------------------------------- #
    # Other
    # -------------------------------------------------------------------------- #
    def reset(self, init_qpos=None):
        """
        Reset the robot to a rest position or a given q-position
        """
        for uid, agent in self.agents_dict.items():
            robot = agent.robot
            if init_qpos is not None and uid in init_qpos:
                robot.set_qpos(init_qpos[uid])
            robot.set_qvel(torch.zeros(robot.max_dof, device=self.device))
            robot.set_qf(torch.zeros(robot.max_dof, device=self.device))
            agent.set_control_mode(agent._default_control_mode)
