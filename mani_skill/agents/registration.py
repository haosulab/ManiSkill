from dataclasses import dataclass
from typing import Dict, List

from mani_skill import logger
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.utils import assets


@dataclass
class AgentSpec:
    agent_cls: type[BaseAgent]
    asset_download_ids: List[str]


REGISTERED_AGENTS: Dict[str, AgentSpec] = {}


def register_agent(asset_download_ids: List[str] = [], override=False):
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

        REGISTERED_AGENTS[agent_cls.uid] = AgentSpec(
            agent_cls=agent_cls, asset_download_ids=asset_download_ids
        )
        assets.DATA_GROUPS[agent_cls.uid] = asset_download_ids
        return agent_cls

    return _register_agent
