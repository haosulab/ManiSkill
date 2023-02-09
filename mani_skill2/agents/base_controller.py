from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import sapien.core as sapien
from gym import spaces

from mani_skill2.utils.common import clip_and_scale_action, normalize_action_space

from .utils import flatten_action_spaces, get_active_joint_indices, get_active_joints


class BaseController:
    """Base class for controllers.
    The controller is an interface for the robot to interact with the environment.
    """

    joints: List[sapien.Joint]  # active joints controlled
    joint_indices: List[int]  # indices of active joints controlled
    action_space: spaces.Space

    def __init__(
        self,
        config: "ControllerConfig",
        articulation: sapien.Articulation,
        control_freq: int,
        sim_freq: int = None,
    ):
        self.config = config
        self.articulation = articulation
        self._control_freq = control_freq

        # For action interpolation
        if sim_freq is None:  # infer from scene
            # TODO(jigu): update sapien interface to avoid this workaround
            sim_timestep = self.articulation.get_builder().get_scene().get_timestep()
            sim_freq = round(1.0 / sim_timestep)
        # Number of simulation steps per control step
        self._sim_steps = sim_freq // control_freq

        self._initialize_joints()
        self._initialize_action_space()

        # NOTE(jigu): It is intended not to be a required field in config.
        self._normalize_action = getattr(self.config, "normalize_action", False)
        if self._normalize_action:
            self._clip_and_scale_action_space()

    def _initialize_joints(self):
        joint_names = self.config.joint_names
        try:
            self.joints = get_active_joints(self.articulation, joint_names)
            self.joint_indices = get_active_joint_indices(
                self.articulation, joint_names
            )
        except Exception as err:
            print("Encounter error when parsing joint names.")
            active_joint_names = [x.name for x in self.articulation.get_active_joints()]
            print("Joint names of the articulation", active_joint_names)
            print("Joint names of the controller", joint_names)
            raise err

    def _initialize_action_space(self):
        # self.action_space = spaces.Box(...)
        raise NotImplementedError

    @property
    def control_freq(self):
        return self._control_freq

    @property
    def qpos(self):
        """Get current joint positions."""
        return self.articulation.get_qpos()[self.joint_indices]

    @property
    def qvel(self):
        """Get current joint velocities."""
        return self.articulation.get_qvel()[self.joint_indices]

    # -------------------------------------------------------------------------- #
    # Interfaces (implemented in subclasses)
    # -------------------------------------------------------------------------- #
    def set_drive_property(self):
        """Set the joint drive property according to the config."""
        raise NotImplementedError

    def reset(self):
        """Called after switching the controller."""
        self.set_drive_property()

    def _preprocess_action(self, action: np.ndarray):
        # TODO(jigu): support discrete action
        action_dim = self.action_space.shape[0]
        assert action.shape == (action_dim,), (action.shape, action_dim)
        if self._normalize_action:
            action = self._clip_and_scale_action(action)
        return action

    def set_action(self, action: np.ndarray):
        """Set the action to execute.
        The action can be low-level control signals or high-level abstract commands.
        """
        raise NotImplementedError

    def before_simulation_step(self):
        """Called before each simulation step in one control step."""
        pass

    def get_state(self) -> dict:
        """Get the controller state."""
        return {}

    def set_state(self, state: dict):
        pass

    # -------------------------------------------------------------------------- #
    # Normalize action
    # -------------------------------------------------------------------------- #
    def _clip_and_scale_action_space(self):
        self._action_space = self.action_space
        self.action_space = normalize_action_space(self._action_space)

    def _clip_and_scale_action(self, action):
        return clip_and_scale_action(
            action, self._action_space.low, self._action_space.high
        )


@dataclass
class ControllerConfig:
    joint_names: List[str]
    # NOTE(jigu): It is a class variable in this base class,
    # but you can inherit it and overwrite with an instance variable.
    controller_cls = BaseController


# -------------------------------------------------------------------------- #
# Composite controllers
# -------------------------------------------------------------------------- #
class DictController(BaseController):
    def __init__(
        self,
        configs: Dict[str, ControllerConfig],
        articulation: sapien.Articulation,
        control_freq: int,
        sim_freq: int = None,
        balance_passive_force=True,
    ):
        self.configs = configs
        self.articulation = articulation
        self._control_freq = control_freq
        self.balance_passive_force = balance_passive_force

        self.controllers: Dict[str, BaseController] = OrderedDict()
        for uid, config in configs.items():
            self.controllers[uid] = config.controller_cls(
                config, articulation, control_freq, sim_freq=sim_freq
            )

        self._initialize_action_space()
        self._initialize_joints()
        self._assert_fully_actuated()

    def _initialize_action_space(self):
        # Explicitly create a list of key-value tuples
        # Otherwise, spaces.Dict will sort keys if a dict is provided
        named_spaces = [
            (uid, controller.action_space)
            for uid, controller in self.controllers.items()
        ]
        self.action_space = spaces.Dict(named_spaces)

    def _initialize_joints(self):
        self.joints = []
        self.joint_indices = []
        for controller in self.controllers.values():
            self.joints.extend(controller.joints)
            self.joint_indices.extend(controller.joint_indices)

    def _assert_fully_actuated(self):
        active_joints = self.articulation.get_active_joints()
        if len(active_joints) != len(self.joints) or set(active_joints) != set(
            self.joints
        ):
            print("active_joints:", [x.name for x in active_joints])
            print("controlled_joints:", [x.name for x in self.joints])
            raise AssertionError("{} is not fully actuated".format(self.articulation))

    def set_drive_property(self):
        raise RuntimeError(
            "Undefined behaviors to set drive property for multiple controllers"
        )

    def reset(self):
        for controller in self.controllers.values():
            controller.reset()

    def set_action(self, action: Dict[str, np.ndarray]):
        for uid, controller in self.controllers.items():
            controller.set_action(action[uid])

    def before_simulation_step(self):
        if self.balance_passive_force:
            qf = self.articulation.compute_passive_force(external=False)
        else:
            qf = np.zeros(self.articulation.dof)
        for controller in self.controllers.values():
            ret = controller.before_simulation_step()
            if ret is not None and "qf" in ret:
                qf = qf + ret["qf"]
        self.articulation.set_qf(qf)

    def get_state(self) -> dict:
        states = {}
        for uid, controller in self.controllers.items():
            state = controller.get_state()
            if len(state) > 0:
                states[uid] = state
        return states

    def set_state(self, state: dict):
        for uid, controller in self.controllers.items():
            controller.set_state(state.get(uid))


class CombinedController(DictController):
    """A flat/combined view of multiple controllers."""

    def _initialize_action_space(self):
        super()._initialize_action_space()
        self.action_space, self.action_mapping = flatten_action_spaces(
            self.action_space.spaces
        )

    def set_action(self, action: np.ndarray):
        # Sanity check
        action_dim = self.action_space.shape[0]
        assert action.shape == (action_dim,), (action.shape, action_dim)

        for uid, controller in self.controllers.items():
            start, end = self.action_mapping[uid]
            controller.set_action(action[start:end])

    def to_action_dict(self, action: np.ndarray):
        """Convert a flat action to a dict of actions."""
        # Sanity check
        action_dim = self.action_space.shape[0]
        assert action.shape == (action_dim,), (action.shape, action_dim)

        action_dict = {}
        for uid, controller in self.controllers.items():
            start, end = self.action_mapping[uid]
            action_dict[uid] = action[start:end]
        return action_dict

    def from_action_dict(self, action_dict: dict):
        """Convert a dict of actions to a flat action."""
        return np.hstack([action_dict[uid] for uid in self.controllers])
