from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import sapien

if TYPE_CHECKING:
    from mani_skill.envs.sapien_env import BaseEnv

from mani_skill.utils.structs.actor import Actor


class SceneBuilder:
    """Base class for defining scene builders that can be reused across tasks"""

    env: BaseEnv
    _scene_objects: List[Actor] = []
    _movable_objects: List[Actor] = []
    builds_lighting: bool = False
    """Whether this scene builder will add it's own lighting when build is called. If False, ManiSkill will add some default lighting"""
    scene_configs: List[Any] = None
    """List of scene configuration information that can be used to construct scenes. Can be simply a path to a json file or a dictionary"""

    def __init__(self, env, robot_init_qpos_noise=0.02):
        self.env = env
        self.robot_init_qpos_noise = robot_init_qpos_noise

    def build(self, **kwargs):
        """
        Should create actor/articulation builders and only build objects into the scene without initializing pose, qpos, velocities etc.
        """
        raise NotImplementedError()

    def initialize(self, **kwargs):
        """
        Should initialize the scene, which can include e.g. setting the pose of all objects, changing the qpos/pose of articulations/robots etc.
        """
        raise NotImplementedError()

    @property
    def scene(self):
        return self.env._scene

    @property
    def scene_objects(self) -> List[Actor]:
        raise NotImplementedError()

    @property
    def movable_objects(self) -> List[Actor]:
        raise NotImplementedError()

    @property
    def scene_objects_by_id(self) -> Dict[str, Actor]:
        raise NotImplementedError()

    @property
    def movable_objects_by_id(self) -> Dict[str, Actor]:
        raise NotImplementedError()
