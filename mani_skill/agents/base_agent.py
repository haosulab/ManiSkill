from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import sapien
import torch
from gymnasium import spaces

from mani_skill import format_path
from mani_skill.agents.controllers.pd_joint_pos import (
    PDJointPosController,
    PDJointPosControllerConfig,
)
from mani_skill.sensors.base_sensor import BaseSensor, BaseSensorConfig
from mani_skill.utils import assets, download_asset, sapien_utils
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.structs import Actor, Array, Articulation
from mani_skill.utils.structs.pose import Pose

from .controllers.base_controller import (
    BaseController,
    CombinedController,
    ControllerConfig,
)

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
DictControllerConfig = Dict[str, ControllerConfig]


@dataclass
class Keyframe:
    pose: sapien.Pose
    """sapien Pose object describe this keyframe's pose"""
    qpos: Optional[Array] = None
    """the qpos of the robot at this keyframe"""
    qvel: Optional[Array] = None
    """the qvel of the robot at this keyframe"""


class BaseAgent:
    """Base class for agents/robots, forming an interface of an articulated robot (SAPIEN's physx.PhysxArticulation).
    Users implementing their own agents/robots should inherit from this class.
    A tutorial on how to build your own agent can be found in :doc:`its tutorial </user_guide/tutorials/custom_robots>`

    Args:
        scene (ManiSkillScene): simulation scene instance.
        control_freq (int): control frequency (Hz).
        control_mode (str | None): uid of controller to use
        fix_root_link (bool): whether to fix the robot root link
        agent_idx (str | None): an index for this agent in a multi-agent task setup If None, the task should be single-agent
        initial_pose (sapien.Pose | Pose | None): the initial pose of the robot. Important to set for GPU simulation to ensure robot
        does not collide with other objects in the scene during GPU initialization which occurs before `env._initialize_episode` is called
    """

    uid: str
    """unique identifier string of this"""
    urdf_path: Union[str, None] = None
    """path to the .urdf file describe the agent's geometry and visuals. One of urdf_path or mjcf_path must be provided."""
    urdf_config: Union[str, Dict] = None
    """Optional provide a urdf_config to further modify the created articulation"""
    mjcf_path: Union[str, None] = None
    """path to a MJCF .xml file defining a robot. This will only load the articulation defined in the XML and nothing else.
    One of urdf_path or mjcf_path must be provided."""

    fix_root_link: bool = True
    """Whether to fix the root link of the robot in place."""
    load_multiple_collisions: bool = False
    """Whether the referenced collision meshes of a robot definition should be loaded as multiple convex collisions"""
    disable_self_collisions: bool = False
    """Whether to disable self collisions. This is generally not recommended as you should be defining a SRDF file to exclude specific collisions.
    However for some robots/tasks it may be easier to disable all self collisions between links in the robot to increase simulation speed
    """

    keyframes: Dict[str, Keyframe] = dict()
    """a dict of predefined keyframes similar to what Mujoco does that you can use to reset the agent to that may be of interest"""

    def __init__(
        self,
        scene: ManiSkillScene,
        control_freq: int,
        control_mode: Optional[str] = None,
        agent_idx: Optional[str] = None,
        initial_pose: Optional[Union[sapien.Pose, Pose]] = None,
        build_separate: bool = False,
    ):
        self.scene = scene
        self._control_freq = control_freq
        self._agent_idx = agent_idx
        self.build_separate = build_separate

        self.robot: Articulation = None
        """The robot object, which is an Articulation. Data like pose, qpos etc. can be accessed from this object."""
        self.controllers: Dict[str, BaseController] = dict()
        """The controllers of the robot."""
        self.sensors: Dict[str, BaseSensor] = dict()
        """The sensors that come with the robot."""

        self._load_articulation(initial_pose)
        self._after_loading_articulation()

        # Controller
        self.supported_control_modes = list(self._controller_configs.keys())
        """List of all possible control modes for this robot."""
        if control_mode is None:
            control_mode = self.supported_control_modes[0]
        # The control mode after reset for consistency
        self._default_control_mode = control_mode
        self.set_control_mode()

        self._after_init()

    @property
    def _sensor_configs(self) -> List[BaseSensorConfig]:
        """Returns a list of sensor configs for this agent. By default this is empty."""
        return []

    @property
    def _controller_configs(
        self,
    ) -> Dict[str, Union[ControllerConfig, DictControllerConfig]]:
        """Returns a dict of controller configs for this agent. By default this is a PDJointPos (delta and non delta) controller for all active joints."""
        return dict(
            pd_joint_pos=PDJointPosControllerConfig(
                [x.name for x in self.robot.active_joints],
                lower=None,
                upper=None,
                stiffness=100,
                damping=10,
                normalize_action=False,
            ),
            pd_joint_delta_pos=PDJointPosControllerConfig(
                [x.name for x in self.robot.active_joints],
                lower=-0.1,
                upper=0.1,
                stiffness=100,
                damping=10,
                normalize_action=True,
                use_delta=True,
            ),
        )

    @property
    def device(self):
        return self.scene.device

    def _load_articulation(
        self, initial_pose: Optional[Union[sapien.Pose, Pose]] = None
    ):
        """
        Loads the robot articulation
        """

        def build_articulation(scene_idxs: Optional[List[int]] = None):
            if self.urdf_path is not None:
                loader = self.scene.create_urdf_loader()
                asset_path = format_path(str(self.urdf_path))
            elif self.mjcf_path is not None:
                loader = self.scene.create_mjcf_loader()
                asset_path = format_path(str(self.mjcf_path))

            loader.name = self.uid
            if self._agent_idx is not None:
                loader.name = f"{self.uid}-agent-{self._agent_idx}"
            loader.fix_root_link = self.fix_root_link
            loader.load_multiple_collisions_from_file = self.load_multiple_collisions
            loader.disable_self_collisions = self.disable_self_collisions

            if self.urdf_config is not None:
                urdf_config = sapien_utils.parse_urdf_config(self.urdf_config)
                sapien_utils.check_urdf_config(urdf_config)
                sapien_utils.apply_urdf_config(loader, urdf_config)

            if not os.path.exists(asset_path):
                print(f"Robot {self.uid} definition file not found at {asset_path}")
                if (
                    self.uid in assets.DATA_GROUPS
                    or len(assets.DATA_GROUPS[self.uid]) > 0
                ):
                    response = download_asset.prompt_yes_no(
                        f"Robot {self.uid} has assets available for download. Would you like to download them now?"
                    )
                    if response:
                        for (
                            asset_id
                        ) in assets.expand_data_group_into_individual_data_source_ids(
                            self.uid
                        ):
                            download_asset.download(assets.DATA_SOURCES[asset_id])
                    else:
                        print(
                            f"Exiting as assets for robot {self.uid} are not downloaded"
                        )
                        exit()
                else:
                    print(
                        f"Exiting as assets for robot {self.uid} are not found. Check that this agent is properly registered with the appropriate download asset ids"
                    )
                    exit()
            builder = loader.parse(asset_path)["articulation_builders"][0]
            builder.initial_pose = initial_pose
            if scene_idxs is not None:
                builder.set_scene_idxs(scene_idxs)
                builder.set_name(f"{self.uid}-agent-{self._agent_idx}-{scene_idxs}")
            robot = builder.build()
            assert robot is not None, f"Fail to load URDF/MJCF from {asset_path}"
            return robot

        if self.build_separate:
            arts = []
            for scene_idx in range(self.scene.num_envs):
                robot = build_articulation([scene_idx])
                self.scene.remove_from_state_dict_registry(robot)
                arts.append(robot)
            self.robot = Articulation.merge(
                arts, name=f"{self.uid}-agent-{self._agent_idx}", merge_links=True
            )
            self.scene.add_to_state_dict_registry(self.robot)
        else:
            self.robot = build_articulation()
        # Cache robot link names
        self.robot_link_names = [link.name for link in self.robot.get_links()]

    def _after_loading_articulation(self):
        """Called after loading articulation and before setting up any controllers. By default this is empty."""

    def _after_init(self):
        """Code that is run after initialization. Some example robot implementations use this to cache a reference to special
        robot links like an end-effector link. By default this is empty."""

    # -------------------------------------------------------------------------- #
    # Controllers
    # -------------------------------------------------------------------------- #

    @property
    def control_mode(self):
        """Get the currently activated controller uid."""
        return self._control_mode

    def set_control_mode(self, control_mode: str = None):
        """Sets the controller to an pre-existing controller of this agent.
        This does not reset the controller. If given control mode is None, will set to the default control mode."""
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
            balance_passive_force = True
            if isinstance(config, dict):
                if "balance_passive_force" in config:
                    balance_passive_force = config.pop("balance_passive_force")
                self.controllers[control_mode] = CombinedController(
                    config,
                    self.robot,
                    self._control_freq,
                    scene=self.scene,
                )
            else:
                self.controllers[control_mode] = config.controller_cls(
                    config, self.robot, self._control_freq, scene=self.scene
                )
            self.controllers[control_mode].set_drive_property()
            if balance_passive_force:
                # NOTE (stao): Balancing passive force is currently not supported in PhysX, so we work around by disabling gravity
                if not self.scene._gpu_sim_initialized:
                    for link in self.robot.links:
                        link.disable_gravity = True
                else:
                    for link in self.robot.links:
                        if link.disable_gravity.all() != True:
                            logger.warning(
                                f"Attemped to set control mode and disable gravity for the links of {self.robot}. However the GPU sim has already initialized with the links having gravity enabled so this will not work."
                            )

    @property
    def controller(self) -> BaseController:
        """Get currently activated controller."""
        if self._control_mode is None:
            raise RuntimeError("Please specify a control mode first")
        else:
            return self.controllers[self._control_mode]

    @property
    def action_space(self) -> spaces.Space:
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
    def single_action_space(self) -> spaces.Space:
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
        Set the agent's action which is to be executed in the next environment timestep.
        This is essentially a wrapper around the controller's set_action method.
        """
        if not self.scene.gpu_sim_enabled:
            if np.isnan(action).any():
                raise ValueError("Action cannot be NaN. Environment received:", action)
        self.controller.set_action(action)

    def before_simulation_step(self):
        """Code that runs before each simulation step. By default it calls the controller's before_simulation_step method."""
        self.controller.before_simulation_step()

    # -------------------------------------------------------------------------- #
    # Observations and State
    # -------------------------------------------------------------------------- #
    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent, default is the qpos and qvel of the robot and any controller state.
        """
        obs = dict(qpos=self.robot.get_qpos(), qvel=self.robot.get_qvel())
        controller_state = self.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def get_controller_state(self):
        """
        Get the state of the controller.
        """
        return self.controller.get_state()

    def set_controller_state(self, state: Array):
        """
        Set the state of the controller.
        """
        self.controller.set_state(state)

    def get_state(self) -> Dict:
        """Get current state, including robot state and controller state"""
        state = dict()

        # robot state
        root_link = self.robot.get_links()[0]
        state["robot_root_pose"] = root_link.pose
        state["robot_root_vel"] = root_link.get_linear_velocity()
        state["robot_root_qvel"] = root_link.get_angular_velocity()
        state["robot_qpos"] = self.robot.get_qpos()
        state["robot_qvel"] = self.robot.get_qvel()

        # controller state
        state["controller"] = self.get_controller_state()

        return state

    def set_state(self, state: Dict, ignore_controller=False):
        """Set the state of the agent, including the robot state and controller state.
        If ignore_controller is True, the controller state will not be updated."""
        # robot state
        self.robot.set_root_pose(state["robot_root_pose"])
        self.robot.set_root_linear_velocity(state["robot_root_vel"])
        self.robot.set_root_angular_velocity(state["robot_root_qvel"])
        self.robot.set_qpos(state["robot_qpos"])
        self.robot.set_qvel(state["robot_qvel"])

        if not ignore_controller and "controller" in state:
            self.set_controller_state(state["controller"])
        if self.device.type == "cuda":
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()

    # -------------------------------------------------------------------------- #
    # Other
    # -------------------------------------------------------------------------- #
    def reset(self, init_qpos: torch.Tensor = None):
        """
        Reset the robot to a clean state with zero velocity and forces.

        Args:
            init_qpos (torch.Tensor): The initial qpos to set the robot to. If None, the robot's qpos is not changed.
        """
        if init_qpos is not None:
            self.robot.set_qpos(init_qpos)
        self.robot.set_qvel(torch.zeros(self.robot.max_dof, device=self.device))
        self.robot.set_qf(torch.zeros(self.robot.max_dof, device=self.device))

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
