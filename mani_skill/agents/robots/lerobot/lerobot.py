"""
Code based on https://github.com/huggingface/lerobot for supporting real robot control via the unified LeRobot interface.
"""

from mani_skill.agents.base_real_agent import BaseRealAgent


class LeRobotAgent(BaseRealAgent):
    """
    LeRobotAgent is a class for controlling a real robot.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
