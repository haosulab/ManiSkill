from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


@register_agent()  # uncomment this if you want to register the agent so you can instantiate it by ID when creating environments
class Stompy(BaseAgent):
    uid = "stompy"
    urdf_path = f"{ASSET_DIR}/robots/stompy/robot.urdf"
    urdf_config = dict()
