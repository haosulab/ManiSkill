from dataclasses import dataclass, field
from typing import List

import sapien

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.envs.scene import ManiSkillScene
from mani_skill2.utils.structs.actor import Actor


@dataclass
class SceneBuilder:
    env: BaseEnv
    _scene_objects: List[Actor] = field(default_factory=list)
    _movable_objects: List[Actor] = field(default_factory=list)

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
    def scene_objects(self):
        raise NotImplementedError()

    @property
    def movable_objects(self):
        raise NotImplementedError()
