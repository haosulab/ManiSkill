from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import sapien
import sapien.physx as physx
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from mani_skill.agents.utils import (
    flatten_action_spaces,
    get_active_joint_indices,
    get_joints_by_names,
)
from mani_skill.utils import common, gym_utils
from mani_skill.utils.structs import Articulation, ArticulationJoint
from mani_skill.utils.structs.types import Array

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene


class BaseController:
    """Base class for controllers.
    The controller is an interface for the robot to interact with the environment.
    """

    joints: List[ArticulationJoint]
    """active joints controlled"""
    active_joint_indices: torch.Tensor
    """indices of active joints controlled. Equivalent to [x.active_index for x in self.joints]"""
    action_space: spaces.Space
    """the action space of the controller, which by default has a batch dimension. This is typically already normalized as well."""
    single_action_space: spaces.Space
    """The unbatched version of the action space which is also typically already normalized by this class"""
    _original_single_action_space: spaces.Space
    """The unbatched, original action space without any additional processing like normalization"""

    sets_target_qpos: bool
    """Whether the controller sets the target qpos of the articulation"""
    sets_target_qvel: bool
    """Whether the controller sets the target qvel of the articulation"""

    def __init__(
        self,
        config: "ControllerConfig",
        articulation: Articulation,
        control_freq: int,
        sim_freq: Optional[int] = None,
        scene: ManiSkillScene = None,
    ):
        self.config = config
        self.articulation = articulation
        self._control_freq = control_freq
        self.scene = scene

        # For action interpolation
        if sim_freq is None:  # infer from scene
            sim_timestep = self.articulation.px.timestep
            sim_freq = round(1.0 / sim_timestep)
        self._sim_steps = sim_freq // control_freq
        """Number of simulation steps per control step"""

        self._initialize_joints()
        self._initialize_action_space()
        # NOTE(jigu): It is intended not to be a required field in config.
        self._normalize_action = getattr(self.config, "normalize_action", False)
        if self._normalize_action:
            self._clip_and_scale_action_space()
        self.action_space = self.single_action_space
        if self.scene.num_envs > 1:
            self.action_space = batch_space(
                self.single_action_space, n=self.scene.num_envs
            )

    @property
    def device(self):
        return self.articulation.device

    def _initialize_joints(self):
        joint_names = self.config.joint_names
        try:
            # We only track the joints we can control, the active ones.
            self.joints = get_joints_by_names(self.articulation, joint_names)
            self.active_joint_indices = get_active_joint_indices(
                self.articulation, joint_names
            ).to(self.device)
        except Exception as err:
            print("Encounter error when parsing joint names.")
            active_joint_names = [x.name for x in self.articulation.get_active_joints()]
            print("Joint names of the articulation", active_joint_names)
            print("Joint names of the controller", joint_names)
            raise err

    def _initialize_action_space(self):
        raise NotImplementedError

    @property
    def control_freq(self):
        return self._control_freq

    @property
    def qpos(self):
        """Get current joint positions."""
        return self.articulation.get_qpos()[..., self.active_joint_indices]

    @property
    def qvel(self):
        """Get current joint velocities."""
        return self.articulation.get_qvel()[..., self.active_joint_indices]

    # -------------------------------------------------------------------------- #
    # Interfaces (implemented in subclasses)
    # -------------------------------------------------------------------------- #
    def set_drive_property(self):
        """Set the joint drive property according to the config."""
        raise NotImplementedError

    def reset(self):
        """Resets the controller to an initial state. This is called upon environment creation and each environment reset"""

    def _preprocess_action(self, action: Array):
        # TODO(jigu): support discrete action
        if self.scene.num_envs > 1:
            action_dim = self.action_space.shape[1]
        else:
            action_dim = self.action_space.shape[0]
        assert action.shape == (self.scene.num_envs, action_dim), (
            action.shape,
            action_dim,
        )

        if self._normalize_action:
            action = self._clip_and_scale_action(action)
        return action

    def set_action(self, action: Array):
        """Set the action to execute.
        The action can be low-level control signals or high-level abstract commands.
        """
        raise NotImplementedError

    def before_simulation_step(self):
        """Called before each simulation step in one control step."""

    def get_state(self) -> dict:
        """Get the controller state."""
        return {}

    def set_state(self, state: dict):
        pass

    # -------------------------------------------------------------------------- #
    # Normalize action
    # -------------------------------------------------------------------------- #
    def _clip_and_scale_action_space(self):
        self._original_single_action_space = self.single_action_space
        self.single_action_space = gym_utils.normalize_action_space(
            self._original_single_action_space
        )
        low, high = (
            self._original_single_action_space.low,
            self._original_single_action_space.high,
        )
        self.action_space_low = common.to_tensor(low, device=self.device)
        self.action_space_high = common.to_tensor(high, device=self.device)

    def _clip_and_scale_action(self, action):
        return gym_utils.clip_and_scale_action(
            action, self.action_space_low, self.action_space_high
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(dof={len(self.active_joint_indices)}, active_joints={len(self.joints)}, joints=({', '.join([x.name for x in self.joints])}))"


@dataclass
class ControllerConfig:
    joint_names: List[str]
    """the names of the joints to control. Note that some controller configurations might not actually let you directly control all the given joints
    and will instead have some other implicit control (e.g. Passive controllers or mimic controllers)."""
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
        articulation: Articulation,
        control_freq: int,
        sim_freq: int = None,
        scene: ManiSkillScene = None,
    ):
        self.scene = scene
        self.configs = configs
        self.articulation = articulation
        self._control_freq = control_freq

        self.controllers: Dict[str, BaseController] = dict()
        for uid, config in configs.items():
            self.controllers[uid] = config.controller_cls(
                config, articulation, control_freq, sim_freq=sim_freq, scene=scene
            )
        self._initialize_action_space()
        self._initialize_joints()

        self.action_space = self.single_action_space
        if self.scene.num_envs > 1:
            self.action_space = batch_space(
                self.single_action_space, n=self.scene.num_envs
            )

        self.sets_target_qpos = False
        self.sets_target_qvel = False
        for controller in self.controllers.values():
            self.sets_target_qpos = self.sets_target_qpos or controller.sets_target_qpos
            self.sets_target_qvel = self.sets_target_qvel or controller.sets_target_qvel

    def before_simulation_step(self):
        for controller in self.controllers.values():
            controller.before_simulation_step()

    def _initialize_action_space(self):
        # Explicitly create a list of key-value tuples
        # Otherwise, spaces.Dict will sort keys if a dict is provided
        named_spaces = [
            (uid, controller.single_action_space)
            for uid, controller in self.controllers.items()
        ]
        self.single_action_space = spaces.Dict(named_spaces)

    def _initialize_joints(self):
        self.joints = []
        self.active_joint_indices = []
        for controller in self.controllers.values():
            self.joints.extend(controller.joints)
            self.active_joint_indices.extend(controller.active_joint_indices)

    def set_drive_property(self):
        for controller in self.controllers.values():
            controller.set_drive_property()

    def reset(self):
        for controller in self.controllers.values():
            controller.reset()

    def set_action(self, action: Dict[str, np.ndarray]):
        for uid, controller in self.controllers.items():
            controller.set_action(action[uid])

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

    def from_qpos(self, qpos: Array):
        """Tries to generate the corresponding action given a full robot qpos.
        This can be useful for joint position control when setting a desired qposition even
        if some controllers merge some joints together like the mimic controller
        """
        qpos = common.to_tensor(qpos, device=self.device)
        if len(qpos.shape) > 1:
            assert qpos.shape[1] == len(self.joints)
        else:
            assert len(qpos) == len(self.joints)
        final_action = []
        start = 0
        for controller in self.controllers.values():
            # if (controller, PDJointPosMimicController)
            ndims = controller.single_action_space.shape[0]
            njoints = len(controller.joints)
            sub_action = qpos[..., start : start + ndims]
            start = start + njoints
            final_action.append(sub_action)
        return torch.concat(final_action)

    def __repr__(self):
        dims = len(self.joints)
        dof = sum(
            [
                controller.single_action_space.shape[0]
                for controller in self.controllers.values()
            ]
        )
        main_str = f"{self.__class__.__name__}(dof={dof}, active_joints={dims}"
        for uid, controller in self.controllers.items():
            controller_str = controller.__repr__()
            controller_str = controller_str.replace("\n", "\n    ")
            main_str += f"\n    {uid}: {controller_str}"
        return main_str + "\n)"


class CombinedController(DictController):

    """A flat/combined view of multiple controllers."""

    def _initialize_action_space(self):
        super()._initialize_action_space()
        self.single_action_space, self.action_mapping = flatten_action_spaces(
            self.single_action_space.spaces
        )

    def set_action(self, action: np.ndarray):
        # Sanity check
        # TODO (stao): optimization, do we really need this sanity check? Does gymnasium already do this for us
        if self.scene.num_envs > 1:
            action_dim = self.action_space.shape[1]
        else:
            action_dim = self.action_space.shape[0]
        assert action.shape == (
            self.scene.num_envs,
            action_dim,
        ), f"Received action of shape {action.shape} but expected shape ({self.scene.num_envs}, {action_dim})"
        for uid, controller in self.controllers.items():
            start, end = self.action_mapping[uid]
            controller.set_action(action[:, start:end])

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
        return torch.hstack([action_dict[uid] for uid in self.controllers])
