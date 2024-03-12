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

from mani_skill import logger
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.articulation import Articulation

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene


class ArticulationBuilder(SapienArticulationBuilder):
    scene: ManiSkillScene
    disable_self_collisions: bool = False

    def __init__(self):
        super().__init__()
        self.name = None
        self.scene_idxs = None

    def set_name(self, name: str):
        self.name = name
        return self

    def set_scene_idxs(
        self,
        scene_idxs: Optional[
            Union[List[int], Sequence[int], torch.Tensor, np.ndarray]
        ] = None,
    ):
        """
        Set a list of scene indices to build this object in. Cannot be used in conjunction with scene mask
        """
        self.scene_idxs = scene_idxs
        return self

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

    def build(self, name=None, fix_root_link=None, build_mimic_joints=True):
        assert self.scene is not None
        if name is not None:
            self.set_name(name)
        assert self.name is not None
        if self.scene_idxs is not None:
            pass
        else:
            self.scene_idxs = torch.arange((self.scene.num_envs), dtype=int)

        articulations = []

        for scene_idx in self.scene_idxs:
            sub_scene = self.scene.sub_scenes[scene_idx]
            links: List[sapien.Entity] = self.build_entities(
                name_prefix=f"scene-{scene_idx}-{self.name}_"
            )
            if fix_root_link is not None:
                links[0].components[0].joint.type = (
                    "fixed" if fix_root_link else "undefined"
                )
            links[0].pose = self.initial_pose

            articulation = links[0].components[0].articulation
            if build_mimic_joints:
                for mimic in self.mimic_joint_records:
                    joint = articulation.find_joint_by_name(
                        f"scene-{scene_idx}-{self.name}_{mimic.joint}"
                    )
                    mimic_joint = articulation.find_joint_by_name(
                        f"scene-{scene_idx}-{self.name}_{mimic.mimic}"
                    )
                    multiplier = mimic.multiplier
                    offset = mimic.offset

                    # joint mimics parent
                    if joint.parent_link == mimic_joint.child_link:
                        if joint.parent_link.parent is None:
                            logger.warn(
                                f"Skipping adding fixed tendon for {joint.name}"
                            )
                            # tendon must be attached to grandparent
                            continue
                        root = joint.parent_link.parent
                        parent = joint.parent_link
                        child = joint.child_link
                        articulation.create_fixed_tendon(
                            [root, parent, child],
                            [0, -multiplier, 1],
                            [0, -1 / multiplier, 1],
                            rest_length=offset,
                            stiffness=1e5,
                        )
                    # 2 children mimic each other
                    if joint.parent_link == mimic_joint.parent_link:
                        assert joint.parent_link is not None
                        root = joint.parent_link
                        articulation.create_fixed_tendon(
                            [root, joint.child_link, mimic_joint.child_link],
                            [0, -multiplier, 1],
                            [0, -1 / multiplier, 1],
                            rest_length=offset,
                            stiffness=1e5,
                        )

            for l in links:
                sub_scene.add_entity(l)
            articulation: physx.PhysxArticulation = l.components[0].articulation
            articulation.name = f"scene-{scene_idx}_{self.name}"
            articulations.append(articulation)

        articulation = Articulation.create_from_physx_articulations(
            articulations, self.scene, self.scene_idxs
        )
        self.scene.articulations[self.name] = articulation
        return articulation
