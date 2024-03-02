from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch
from sapien.wrapper.articulation_builder import (
    ArticulationBuilder as SapienArticulationBuilder,
)
from sapien.wrapper.articulation_builder import LinkBuilder

from mani_skill2.utils.sapien_utils import to_tensor
from mani_skill2.utils.structs.articulation import Articulation

if TYPE_CHECKING:
    from mani_skill2.envs.scene import ManiSkillScene


class ArticulationBuilder(SapienArticulationBuilder):
    scene: ManiSkillScene
    disable_self_collisions: bool = False

    def __init__(self):
        super().__init__()
        self.scene_mask = None
        self.name = None

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

    def create_link_builder(self, parent: LinkBuilder = None):
        if self.link_builders:
            assert parent and parent in self.link_builders

        builder = LinkBuilder(len(self.link_builders), parent)
        self.link_builders.append(builder)
        if self.disable_self_collisions:
            builder.collision_groups[2] |= 1 << 29

        return builder

    def build_entities(self, fix_root_link=None, name_prefix=""):
        entities = []
        links = []
        for b in self.link_builders:
            b._check()
            b.physx_body_type = "link"

            entity = sapien.Entity()

            link_component = b.build_physx_component(
                links[b.parent.index] if b.parent else None
            )

            entity.add_component(link_component)
            if b.visual_records:
                entity.add_component(b.build_render_component())
            entity.name = b.name

            link_component.name = f"{name_prefix}{b.name}"
            link_component.joint.name = f"{name_prefix}{b.joint_record.name}"
            link_component.joint.type = b.joint_record.joint_type
            link_component.joint.pose_in_child = b.joint_record.pose_in_child
            link_component.joint.pose_in_parent = b.joint_record.pose_in_parent

            if link_component.joint.type in [
                "revolute",
                "prismatic",
                "revolute_unwrapped",
            ]:
                link_component.joint.limit = np.array(b.joint_record.limits).flatten()
                link_component.joint.set_drive_property(0, b.joint_record.damping)

            if link_component.joint.type == "continuous":
                link_component.joint.limit = [-np.inf, np.inf]
                link_component.joint.set_drive_property(0, b.joint_record.damping)

            links.append(link_component)
            entities.append(entity)

        if fix_root_link is not None:
            entities[0].components[0].joint.type = (
                "fixed" if fix_root_link else "undefined"
            )
        entities[0].pose = self.initial_pose
        return entities

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

        articulations = []

        for scene_idx, scene in enumerate(self.scene.sub_scenes):
            if self.scene_mask is not None and self.scene_mask[scene_idx] == False:
                continue
            links: List[sapien.Entity] = self.build_entities(
                name_prefix=f"scene-{scene_idx}-{self.name}_"
            )
            if fix_root_link is not None:
                links[0].components[0].joint.type = (
                    "fixed" if fix_root_link else "undefined"
                )
            links[0].pose = self.initial_pose
            for l in links:
                scene.add_entity(l)
            articulation: physx.PhysxArticulation = l.components[0].articulation
            articulation.name = f"scene-{scene_idx}_{self.name}"
            articulations.append(articulation)

        articulation = Articulation._create_from_physx_articulations(
            articulations, self.scene, self.scene_mask
        )
        self.scene.articulations[self.name] = articulation
        return articulation
