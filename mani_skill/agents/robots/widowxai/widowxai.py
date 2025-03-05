import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor


@register_agent(asset_download_ids=["widowxai"])
class WidowXAI(BaseAgent):
    uid = "widowxai"
    urdf_path = f"{ASSET_DIR}/robots/widowxai/wxai.urdf"
    urdf_config = dict()

    arm_joint_names = [
        "joint_0",
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
    ]
    gripper_joint_names = ["left_gripper_joint", "right_gripper_joint"]