from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import numpy as np
import sapien
import sapien.physx as physx
from gymnasium import spaces

from mani_skill2 import format_path
from mani_skill2.sensors.base_sensor import BaseSensor, BaseSensorConfig
from mani_skill2.utils.sapien_utils import (
    apply_urdf_config,
    check_urdf_config,
    parse_urdf_config,
)

from .controllers.base_controller import (
    BaseController,
    CombinedController,
    ControllerConfig,
)


class BaseAgent:
    """Base class for agents.

    Agent is an interface of an articulated robot (physx.PhysxArticulation).

    Args:
        scene (sapien.Scene): simulation scene instance.
        control_freq (int): control frequency (Hz).
        control_mode: uid of controller to use
        fix_root_link: whether to fix the robot root link
        config: agent configuration
    """

    uid: str
    robot: physx.PhysxArticulation

    urdf_path: str
    urdf_config: dict

    controller_configs: Dict[str, Union[ControllerConfig, Dict[str, ControllerConfig]]]
    controllers: Dict[str, BaseController]

    sensor_configs: Dict[str, BaseSensorConfig]
    sensors: Dict[str, BaseSensor]

    def __init__(
        self,
        scene: sapien.Scene,
        control_freq: int,
        control_mode: str = None,
        fix_root_link=True,
    ):
        self.scene = scene
        self._control_freq = control_freq

        # URDF
        self.fix_root_link = fix_root_link

        # Controller
        self.supported_control_modes = list(self.controller_configs.keys())
        if control_mode is None:
            control_mode = self.supported_control_modes[0]
        # The control mode after reset for consistency
        self._default_control_mode = control_mode

        self._load_articulation()
        self._after_loading_articulation()
        self._setup_controllers()
        self.set_control_mode(control_mode)
        self._after_init()

    def _load_articulation(self):
        """
        Load the robot articulation
        """
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = self.fix_root_link

        urdf_path = format_path(str(self.urdf_path))

        urdf_config = parse_urdf_config(self.urdf_config, self.scene)
        check_urdf_config(urdf_config)

        # TODO(jigu): support loading multiple convex collision shapes

        apply_urdf_config(loader, urdf_config)
        self.robot: physx.PhysxArticulation = loader.load(urdf_path)
        assert self.robot is not None, f"Fail to load URDF from {urdf_path}"
        self.robot.set_name(Path(urdf_path).stem)

        # Cache robot link ids
        self.robot_link_ids = [link.name for link in self.robot.get_links()]

    def _after_loading_articulation(self):
        """After loading articulation and before setting up controller. Not recommended, but is useful for when creating
        robot classes that inherit controllers from another and only change which joints are controlled
        """
        pass

    def _after_init(self):
        """After initialization. E.g., caching the end-effector link."""
        pass

    # -------------------------------------------------------------------------- #
    # Controllers
    # -------------------------------------------------------------------------- #

    @property
    def control_mode(self):
        """Get the currently activated controller uid."""
        return self._control_mode

    def set_control_mode(self, control_mode):
        """Set the controller and reset."""
        assert (
            control_mode in self.supported_control_modes
        ), "{} not in supported modes: {}".format(
            control_mode, self.supported_control_modes
        )
        self._control_mode = control_mode
        self.controller.reset()

    def _setup_controllers(self):
        """
        Create and setup the controllers
        """
        self.controllers = OrderedDict()
        for uid, config in self.controller_configs.items():
            if isinstance(config, dict):
                self.controllers[uid] = CombinedController(
                    config, self.robot, self._control_freq
                )
            else:
                self.controllers[uid] = config.controller_cls(
                    config, self.robot, self._control_freq
                )

    @property
    def controller(self):
        """Get currently activated controller."""
        if self._control_mode is None:
            raise RuntimeError("Please specify a control mode first")
        else:
            return self.controllers[self._control_mode]

    @property
    def action_space(self):
        if self._control_mode is None:
            return spaces.Dict(
                {
                    uid: controller.action_space
                    for uid, controller in self.controllers.items()
                }
            )
        else:
            return self.controller.action_space

    def set_action(self, action):
        """
        Set the agent's action which is to be executed in the next environment timestep
        """
        if np.isnan(action).any():
            raise ValueError("Action cannot be NaN. Environment received:", action)
        self.controller.set_action(action)

    def before_simulation_step(self):
        self.controller.before_simulation_step()

    # -------------------------------------------------------------------------- #
    # Observations and State
    # -------------------------------------------------------------------------- #
    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent.
        """
        obs = OrderedDict(qpos=self.robot.get_qpos(), qvel=self.robot.get_qvel())
        controller_state = self.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def get_state(self) -> Dict:
        """Get current state, including robot state and controller state"""
        state = OrderedDict()

        # robot state
        root_link = self.robot.get_links()[0]
        state["robot_root_pose"] = root_link.get_pose()
        state["robot_root_vel"] = root_link.get_linear_velocity()
        state["robot_root_qvel"] = root_link.get_angular_velocity()
        state["robot_qpos"] = self.robot.get_qpos()
        state["robot_qvel"] = self.robot.get_qvel()
        state["robot_qacc"] = self.robot.get_qacc()

        # controller state
        state["controller"] = self.controller.get_state()

        return state

    def set_state(self, state: Dict, ignore_controller=False):
        # robot state
        self.robot.set_root_pose(state["robot_root_pose"])
        self.robot.set_root_velocity(state["robot_root_vel"])
        self.robot.set_root_angular_velocity(state["robot_root_qvel"])
        self.robot.set_qpos(state["robot_qpos"])
        self.robot.set_qvel(state["robot_qvel"])
        self.robot.set_qacc(state["robot_qacc"])

        if not ignore_controller and "controller" in state:
            self.controller.set_state(state["controller"])

    # -------------------------------------------------------------------------- #
    # Other
    # -------------------------------------------------------------------------- #
    def reset(self, init_qpos=None):
        """
        Reset the robot to a rest position or a given q-position
        """
        if init_qpos is not None:
            self.robot.set_qpos(init_qpos)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        self.robot.set_qacc(np.zeros(self.robot.dof))
        self.robot.set_qf(np.zeros(self.robot.dof))
        self.set_control_mode(self._default_control_mode)

    # -------------------------------------------------------------------------- #
    # Optional per-agent APIs, implemented depending on agent affordances
    # -------------------------------------------------------------------------- #
    def is_grasping(self, object: Union[sapien.Entity, None] = None):
        """
        Check if this agent is grasping an object or grasping anything at all

        Args:
            object (sapien.Entity | None):
                If object is a sapien.Entity, this function checks grasping against that. If it is none, the function checks if the
                agent is grasping anything at all.

        Returns:
            True if agent is grasping object. False otherwise. If object is None, returns True if agent is grasping something, False if agent is grasping nothing.
        """
        raise NotImplementedError()
