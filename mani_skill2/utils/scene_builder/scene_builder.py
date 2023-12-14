from typing import List

import sapien


class SceneBuilder:
    def __init__(self):
        self._scene: sapien.Scene = None

        # set in self.build
        self._scene_objects: List[sapien.Entity] = []
        self._movable_objects: List[sapien.Entity] = []

    def build(self, scene, **kwargs):
        raise NotImplementedError()

    @property
    def scene_objects(self):
        raise NotImplementedError()

    @property
    def movable_objects(self):
        raise NotImplementedError()
