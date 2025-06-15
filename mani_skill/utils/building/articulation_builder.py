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
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Articulation, Pose
from mani_skill.utils.structs.pose import to_sapien_pose

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene


class ArticulationBuilder(SapienArticulationBuilder):
    scene: ManiSkillScene
    disable_self_collisions: bool = False

    def __init__(self):
        super().__init__()
        self.name = None
        self.scene_idxs = None
        self.initial_pose = None

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

    def build_entities(self, *args, **kwargs):
        raise NotImplementedError(
            "_build_entities is a private function in ManiSkill. Use build() to properly build articulation"
        )

    def _build_entities(
        self, fix_root_link=None, name_prefix="", initial_pose=sapien.Pose()
    ):
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
        entities[0].pose = initial_pose
        return entities

    def build(
        self, name=None, fix_root_link=None, build_mimic_joints=True
    ) -> Articulation:
        assert self.scene is not None
        if name is not None:
            self.set_name(name)
        assert (
            self.name is not None
            and self.name != ""
            and self.name not in self.scene.articulations
        ), "built actors in ManiSkill must have unique names and cannot be None or empty strings"

        if self.scene_idxs is not None:
            pass
        else:
            self.scene_idxs = torch.arange((self.scene.num_envs), dtype=int)
        num_arts = len(self.scene_idxs)

        if self.initial_pose is None:
            logger.warn(
                f"No initial pose set for articulation builder of {self.name}, setting to default pose q=[1,0,0,0], p=[0,0,0]. There may be simulation issues/bugs if this articulation at it's initial pose collides with other objects at their initial poses."
            )
            self.initial_pose = sapien.Pose()
        self.initial_pose = Pose.create(self.initial_pose)
        initial_pose_b = self.initial_pose.raw_pose.shape[0]
        assert initial_pose_b == 1 or initial_pose_b == num_arts
        initial_pose_np = common.to_numpy(self.initial_pose.raw_pose)

        articulations = []
        for i, scene_idx in enumerate(self.scene_idxs):
            if self.scene.parallel_in_single_scene:
                sub_scene = self.scene.sub_scenes[0]
            else:
                sub_scene = self.scene.sub_scenes[scene_idx]
            if initial_pose_b == 1:
                articulation_pose = to_sapien_pose(initial_pose_np)
            else:
                articulation_pose = to_sapien_pose(initial_pose_np[i])
            links: List[sapien.Entity] = self._build_entities(
                name_prefix=f"scene-{scene_idx}-{self.name}_",
                initial_pose=articulation_pose,
            )
            if fix_root_link is not None:
                links[0].components[0].joint.type = (
                    "fixed" if fix_root_link else "undefined"
                )
            articulation: physx.PhysxArticulation = links[0].components[0].articulation
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
            articulation.pose = articulation_pose
            for l in links:
                sub_scene.add_entity(l)
            articulation.name = f"scene-{scene_idx}_{self.name}"
            articulations.append(articulation)

        articulation: Articulation = Articulation.create_from_physx_articulations(
            articulations, self.scene, self.scene_idxs
        )
        articulation.initial_pose = self.initial_pose
        self.scene.articulations[self.name] = articulation
        self.scene.add_to_state_dict_registry(articulation)
        return articulation
