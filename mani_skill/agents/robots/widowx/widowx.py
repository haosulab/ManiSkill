import numpy as np
import sapien

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


# TODO (stao) (xuanlin): model it properly based on real2sim
@register_agent()
class WidowX250S(BaseAgent):
    uid = "widowx250s"
    urdf_path = f"{ASSET_DIR}/robots/widowx/wx250s.urdf"
    urdf_config = dict()


# class WidowXSinkCameraSetupConfig(WidowXDefaultConfig):
#     @property
#     def cameras(self):
#         return [
#             CameraConfig(
#                 uid="3rd_view_camera",  # the camera used for real evaluation for the sink setup
#                 # p=[0.13, 0.27, 1.24],
#                 # q=look_at([0, 0, 0], [-1, -0.45, -1.05], [0, 0, 1]).q,
#                 # actor_uid=None,
#                 p=[-0.00300001, -0.21, 0.39],
#                 q=[-0.907313, 0.0782, -0.36434, -0.194741],
#                 actor_uid="base_link",
#                 width=640,
#                 height=480,
#                 fov=1.5,  # ignored if intrinsic is passed
#                 near=0.01,
#                 far=10,
#                 intrinsic = np.array([[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]])
#             )
#         ]
