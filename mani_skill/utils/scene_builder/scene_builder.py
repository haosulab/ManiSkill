from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import sapien
import torch
from gymnasium import spaces

from mani_skill.utils.structs.pose import Pose

if TYPE_CHECKING:
    from mani_skill.envs.sapien_env import BaseEnv

from mani_skill.utils.structs import Actor, Articulation
from mani_skill.utils.structs.types import Array


class SceneBuilder:
    """Base class for defining scene builders that can be reused across tasks"""

    env: BaseEnv
    """Env which scenebuilder will build in."""

    robot_init_qpos_noise: float = 0.02
    """Robot init qpos noise"""
    robot_initial_pose: Union[sapien.Pose, Pose] = sapien.Pose()
    """Initial pose of the robot (passed to load_agent)"""

    builds_lighting: bool = False
    """Whether this scene builder will add its own lighting when build is called. If False, ManiSkill will add some default lighting"""

    build_configs: Optional[List[Any]] = None
    """List of scene configuration information that can be used to **build** scenes during reconfiguration (i.e. `env.reset(seed=seed, options=dict(reconfigure=True))`). Can be a dictionary, a path to a json file, or some other data. If a scene needs to load build config data, it will index/sample such build configs from this list."""
    init_configs: Optional[List[Any]] = None
    """List of scene configuration information that can be used to **init** scenes during reconfiguration (i.e. `env.reset()`). Can be a dictionary, a path to a json file, or some other data. If a scene needs to load init config data, it will index/sample such init configs from this list."""

    scene_objects: Optional[Dict[str, Actor]] = None
    """Scene objects are any dynamic, kinematic, or static Actor built by the scene builder. Useful for accessing objects in the scene directly."""
    movable_objects: Optional[Dict[str, Actor]] = None
    """Movable objects are any **dynamic** Actor built by the scene builder. movable_objects is a subset of scene_objects. Can be used to query dynamic objects for e.g. task initialization."""
    articulations: Optional[Dict[str, Articulation]] = None
    """Articulations are any articulation loaded in by the scene builder."""

    navigable_positions: Optional[List[Union[Array, spaces.Box]]] = None
    """Some scenes allow for mobile robots to move through these scene. In this case, a list of navigable positions per env_idx (e.g. loaded from a navmesh) should be provided for easy initialization. Can be a discretized list, range, spaces.Box, etc."""

    def __init__(self, env, robot_init_qpos_noise=0.02):
        self.env = env
        self.robot_init_qpos_noise = robot_init_qpos_noise

    def build(self, build_config_idxs: List[int] = None):
        """
        Should create actor/articulation builders and only build objects into the scene without initializing pose, qpos, velocities etc.
        """
        raise NotImplementedError()

    def initialize(self, env_idx: torch.Tensor, init_config_idxs: List[int] = None):
        """
        Should initialize the scene, which can include e.g. setting the pose of all objects, changing the qpos/pose of articulations/robots etc.
        """
        raise NotImplementedError()

    def sample_build_config_idxs(self) -> List[int]:
        """
        Sample idxs of build configs for easy scene randomization. Should be changed to fit shape of self.build_configs.
        """
        return torch.randint(
            low=0, high=len(self.build_configs), size=(self.env.num_envs,)
        ).tolist()

    def sample_init_config_idxs(self) -> List[int]:
        """
        Sample idxs of init configs for easy scene randomization. Should be changed to fit shape of self.init_configs.
        """
        return torch.randint(
            low=0, high=len(self.init_configs), size=(self.env.num_envs,)
        ).tolist()

    @cached_property
    def build_config_names_to_idxs(self) -> Dict[str, int]:
        return dict((v, i) for i, v in enumerate(self.build_configs))

    @cached_property
    def init_config_names_to_idxs(self) -> Dict[str, int]:
        return dict((v, i) for i, v in enumerate(self.init_configs))

    @property
    def scene(self):
        return self.env.scene
