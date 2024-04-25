from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


# @register_agent() # uncomment this if you want to register the agent so you can instantiate it by ID when creating environments
class TemplateRobot(BaseAgent):
    uid = "todo-give-me-a-name!"
    urdf_path = f"path/to/robot.urdf"  # You can use f"{PACKAGE_ASSET_DIR}" to reference a urdf file in the mani_skill /assets package folder
    urdf_config = dict()

    def __init__(self, *args, **kwargs):
        # useful to store some references to robot parts (links and joints) like so below, which is copied from the Panda robot implementation
        self.arm_joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100

        self.gripper_joint_names = [
            "panda_finger_joint1",
            "panda_finger_joint2",
        ]
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 100

        self.ee_link_name = "panda_hand_tcp"

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        raise NotImplementedError()

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="your_custom_camera_on_this_robot",
                p=[0.0464982, -0.0200011, 0.0360011],
                q=[0, 0.70710678, 0, 0.70710678],
                width=128,
                height=128,
                fov=1.57,
                near=0.01,
                far=100,
                entity_uid="your_mounted_camera",
            )
        ]

    def _after_init(self):
        pass
