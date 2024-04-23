from copy import deepcopy

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
import sapien.physx as physx
from mani_skill.utils import sapien_utils

from mani_skill.agents.controllers.base_controller import (
    BaseController,
    CombinedController,
    ControllerConfig,
    DictController,
)


@register_agent()
class CartPole(BaseAgent):
    uid = "cart_pole"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/cartpole/cartpole.urdf"
    urdf_config = dict()
    sensor_configs = {}

    def __init__(self, *args, **kwargs):
        self.slider_joint_name = "slider"
        self.pole_joint_name = "hinge_1"

        self.joint_names = [self.slider_joint_name, self.pole_joint_name]

        self.slider_joint_stiffness = 5000
        self.slider_joint_damping = 0
        self.slider_joint_force_limit = 1000

        super().__init__(*args, **kwargs)

    def _after_init(self):
        self.slider_joint = self.robot.find_joint_by_name(self.slider_joint_name)
        self.pole_joint = self.robot.find_joint_by_name(self.pole_joint_name)
        self.cart_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "cartpole_cart_cart")
        self.pole_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "cartpole_cart_pole_1")

    @property
    def _controller_configs(self):
        # slider
        slider_joint_pos = PDJointPosControllerConfig(
            [self.slider_joint_name],
            None,
            None,
            self.slider_joint_stiffness,
            self.slider_joint_damping,
            self.slider_joint_force_limit,
            normalize_action=False,
        )
        slider_joint_delta_pos = PDJointPosControllerConfig(
            [self.slider_joint_name],
            -0.1,
            0.1,
            self.slider_joint_stiffness,
            self.slider_joint_damping,
            self.slider_joint_force_limit,
            use_delta=True,
        )
        slider_joint_target_delta_pos = deepcopy(slider_joint_delta_pos)
        slider_joint_target_delta_pos.use_target = True


        controller_configs = dict(
            pd_joint_delta_pos=dict(
                slider=slider_joint_delta_pos
            ),
            pd_joint_pos=dict(arm=slider_joint_pos),
            pd_joint_target_delta_pos=dict(slider=slider_joint_target_delta_pos)
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent.
        """
        obs = super().get_proprioception()
        return obs

    def set_control_mode(self, control_mode=None):
        """Set the controller and drive properties. This does not reset the controller. If given control mode is None, will set defaults"""
        if control_mode is None:
            control_mode = self._default_control_mode
        assert (
                control_mode in self.supported_control_modes
        ), "{} not in supported modes: {}".format(
            control_mode, self.supported_control_modes
        )
        self._control_mode = control_mode
        # create controller on the fly here
        if control_mode not in self.controllers:
            config = self._controller_configs[self._control_mode]
            if isinstance(config, dict):
                self.controllers[control_mode] = CombinedController(
                    config, self.robot, self._control_freq, scene=self.scene, balance_passive_force=False
                )
            else:
                self.controllers[control_mode] = config.controller_cls(
                    config, self.robot, self._control_freq, scene=self.scene, balance_passive_force=False
                )
            self.controllers[control_mode].set_drive_property()
            if (
                    isinstance(self.controllers[control_mode], DictController)
                    and self.controllers[control_mode].balance_passive_force
                    and physx.is_gpu_enabled()
            ):
                # NOTE (stao): Balancing passive force is currently not supported in PhysX, so we work around by disabling gravity
                for link in self.robot.links:
                    link.disable_gravity = True
