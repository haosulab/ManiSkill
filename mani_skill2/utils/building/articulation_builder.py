from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch
from sapien.wrapper.articulation_builder import (
    ArticulationBuilder as SapienArticulationBuilder,
)

from mani_skill2.utils.sapien_utils import to_tensor
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

    def set_scene_mask(
        self,
        scene_mask: Optional[
            Union[List[bool], Sequence[bool], torch.Tensor, np.ndarray]
        ] = None,
    ):
        """
        Set a scene mask so that the articulation builder builds the articulation only in a subset of the environments
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
            self.scene_mask = to_tensor(self.scene_mask)
        else:
            # if scene mask is none, set it here
            self.scene_mask = to_tensor(torch.ones((self.scene.num_envs), dtype=bool))

        parallelized = len(self.scene.sub_scenes) > 1
        articulations = []

        for scene_idx, scene in enumerate(self.scene.sub_scenes):
            if self.scene_mask is not None and self.scene_mask[scene_idx] == False:
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

        articulation = Articulation._create_from_physx_articulations(
            articulations, self.scene, self.scene_mask
        )
        self.scene.articulations[self.name] = articulation
        return articulation
