from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Union, List

import numpy as np
import sapien
import sapien.physx as physx
import torch
from gymnasium import spaces

from mani_skill import format_path
from mani_skill.sensors.base_sensor import BaseSensor, BaseSensorConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs import Actor, Articulation

from .controllers.base_controller import (
    BaseController,
    CombinedController,
    ControllerConfig,
    DictController,
)

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
DictControllerConfig = Dict[str, ControllerConfig]

class BaseAgent:
    """Base class for agents.

    Agent is an interface of an articulated robot (physx.PhysxArticulation).

    Args:
        scene (sapien.Scene): simulation scene instance.
        control_freq (int): control frequency (Hz).
        control_mode: uid of controller to use
        fix_root_link: whether to fix the robot root link
        config: agent configuration
        agent_idx: an index for this agent in a multi-agent task setup If None, the task should be single-agent
    """

    uid: str
    """unique identifier string of this"""
    urdf_path: str
    """path to the .urdf file describe the agent's geometry and visuals"""
    urdf_config: dict = None
    """Optional provide a urdf_config to further modify the created articulation"""
    
    def __init__(
        self,
        scene: ManiSkillScene,
        control_freq: int,
        control_mode: str = None,
        fix_root_link=True,
        agent_idx: int = None,
    ):
        self.scene = scene
        self._control_freq = control_freq
        self._agent_idx = agent_idx

        # URDF
        self.fix_root_link = fix_root_link
        
        self.robot: Articulation = None
        self.controllers: Dict[str, BaseController] = dict()
        self.sensors: Dict[str, BaseSensor] = dict()

        # Controller
        self.supported_control_modes = list(self._controller_configs.keys())
        if control_mode is None:
            control_mode = self.supported_control_modes[0]
        # The control mode after reset for consistency
        self._default_control_mode = control_mode
        self.controllers = OrderedDict()
        self._load_articulation()
        self._after_loading_articulation()
        self._after_init()
        self.set_control_mode()
        
    @property
    def _sensor_configs(self) -> List[str, BaseSensorConfig]:
        return []

    @property
    def _controller_configs(self) -> Dict[str, Union[ControllerConfig, DictControllerConfig]]:
        raise NotImplementedError()

    @property
    def device(self):
        return self.scene.device

    def _load_articulation(self):
        """
        Load the robot articulation
        """
        loader = self.scene.create_urdf_loader()
        loader.name = self.uid
        if self._agent_idx is not None:
            loader.name = f"{self.uid}-agent-{self._agent_idx}"
        loader.fix_root_link = self.fix_root_link

        urdf_path = format_path(str(self.urdf_path))

        if self.urdf_config is not None:
            urdf_config = sapien_utils.parse_urdf_config(self.urdf_config, self.scene)
            sapien_utils.check_urdf_config(urdf_config)

        # TODO(jigu): support loading multiple convex collision shapes
        sapien_utils.apply_urdf_config(loader, urdf_config)
        self.robot: Articulation = loader.load(urdf_path)
        assert self.robot is not None, f"Fail to load URDF from {urdf_path}"

        # Cache robot link ids
        self.robot_link_ids = [link.name for link in self.robot.get_links()]

    def _after_loading_articulation(self):
        """After loading articulation and before setting up controller. Not recommended, but is useful for when creating
        robot classes that inherit controllers from another and only change which joints are controlled
        """

    def _after_init(self):
        """After initialization. E.g., caching the end-effector link."""

    # -------------------------------------------------------------------------- #
    # Controllers
    # -------------------------------------------------------------------------- #

    @property
    def control_mode(self):
        """Get the currently activated controller uid."""
        return self._control_mode

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
                    config, self.robot, self._control_freq, scene=self.scene
                )
            else:
                self.controllers[control_mode] = config.controller_cls(
                    config, self.robot, self._control_freq, scene=self.scene
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

    @property
    def single_action_space(self):
        if self._control_mode is None:
            return spaces.Dict(
                {
                    uid: controller.single_action_space
                    for uid, controller in self.controllers.items()
                }
            )
        else:
            return self.controller.single_action_space

    def set_action(self, action):
        """
        Set the agent's action which is to be executed in the next environment timestep
        """
        if not physx.is_gpu_enabled():
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
        self.robot.set_qvel(torch.zeros(self.robot.max_dof, device=self.device))
        self.robot.set_qf(torch.zeros(self.robot.max_dof, device=self.device))
        self.controller.reset()

    # -------------------------------------------------------------------------- #
    # Optional per-agent APIs, implemented depending on agent affordances
    # -------------------------------------------------------------------------- #
    def is_grasping(self, object: Union[Actor, None] = None):
        """
        Check if this agent is grasping an object or grasping anything at all

        Args:
            object (Actor | None):
                If object is a Actor, this function checks grasping against that. If it is none, the function checks if the
                agent is grasping anything at all.

        Returns:
            True if agent is grasping object. False otherwise. If object is None, returns True if agent is grasping something, False if agent is grasping nothing.
        """
        raise NotImplementedError()

    def is_static(self, threshold: float):
        """
        Check if this robot is static (within the given threshold) in terms of the q velocity

        Args:
            threshold (float): The threshold before this agent is considered static

        Returns:
            True if agent is static within the threshold. False otherwise
        """
        raise NotImplementedError()
