from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
import sapien
import sapien.physx as physx
from sapien.wrapper.articulation_builder import (
    ArticulationBuilder as SapienArticulationBuilder,
)

from mani_skill2.utils.structs.articulation import Articulation

if TYPE_CHECKING:
    from mani_skill2.envs.scene import ManiSkillScene


class ArticulationBuilder(SapienArticulationBuilder):
    scene: ManiSkillScene

    def __init__(self):
        super().__init__()
        self.scene_mask = None

    def set_name(self, name: str):
        self.name = name
        return self

    def set_scene_mask(self, scene_mask: Optional[List[bool]] = None):
        """
        Set a scene mask so that the actor builder builds the actor only in a subset of the environments
        """
        self.scene_mask = scene_mask

    def build(self, name=None, fix_root_link=None):
        assert self.scene is not None
        if name is not None:
            self.set_name(name)
        assert self.name is not None

        num_arts = self.scene.num_envs
        if self.scene_mask is not None:
            assert (
                len(self.scene_mask) == self.scene.num_envs
            ), "Scene mask size is not correct. Must be the same as the number of sub scenes"
            num_arts = np.sum(num_arts)

        parallelized = len(self.scene.sub_scenes) > 1
        articulations = []

        i = 0
        for scene_idx, scene in enumerate(self.scene.sub_scenes):
            if self.scene_mask is not None and self.scene_mask[i] == False:
                continue
            links: List[sapien.Entity] = self.build_entities()
            if fix_root_link is not None:
                links[0].components[0].joint.type = (
                    "fixed" if fix_root_link else "undefined"
                )
            links[0].pose = self.initial_pose
            for l in links:
                if parallelized:
                    l.name = f"scene-{scene_idx}_{l.name}"
                scene.add_entity(l)
            articulation: physx.PhysxArticulation = l.components[0].articulation
            if parallelized:
                articulation.name = f"scene-{scene_idx}_{self.name}"
            else:
                articulation.name = f"{self.name}"
            articulations.append(articulation)

            i += 1
        articulation = Articulation.create_from_physx_articulations(articulations)
        self.scene.articulations[self.name] = articulation
        return articulation
