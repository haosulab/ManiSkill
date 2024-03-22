from dataclasses import dataclass
from typing import Dict

from mani_skill import logger
from mani_skill.agents.base_agent import BaseAgent


@dataclass
class AgentSpec:
    """Agent specifications. At the moment it is a simple wrapper around the agent_cls but the dataclass is used in case we may need additional metadata"""

    agent_cls: type[BaseAgent]


REGISTERED_AGENTS: Dict[str, AgentSpec] = {}


def register_agent(override=False):
    """A decorator to register agents into ManiSkill so they can be used easily by string uid.

    Args:
        uid (str): unique id of the agent.
        override (bool): whether to override the agent if it is already registered.
    """

    def _register_agent(agent_cls: type[BaseAgent]):
        if agent_cls.uid in REGISTERED_AGENTS:
            if override:
                logger.warn(f"Overriding registered agent {agent_cls.uid}")
                REGISTERED_AGENTS.pop(agent_cls.uid)
            else:
                logger.warn(
                    f"Agent {agent_cls.uid} is already registered. Skip registration."
                )
            return agent_cls

        REGISTERED_AGENTS[agent_cls.uid] = AgentSpec(agent_cls=agent_cls)
        return agent_cls

    return _register_agent
