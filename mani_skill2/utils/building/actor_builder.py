from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
import sapien.physx as physx
from sapien import ActorBuilder as SAPIENActorBuilder

from mani_skill2.utils.sapien_utils import to_numpy
from mani_skill2.utils.structs.actor import Actor
from mani_skill2.utils.structs.pose import Pose, to_sapien_pose

if TYPE_CHECKING:
    from mani_skill2.envs.scene import ManiSkillScene


class ActorBuilder(SAPIENActorBuilder):
    """
    ActorBuilder class to flexibly build actors in both CPU and GPU simulations.
    This directly inherits the original flexible ActorBuilder from sapien and changes the build functions to support a batch of scenes and return a batch of Actors
    """

    scene: ManiSkillScene

    def __init__(self):
        super().__init__()
        self.initial_pose = Pose.create(self.initial_pose)
        self.scene_mask = None

    def set_scene_mask(self, scene_mask: Optional[List[bool]] = None):
        """
        Set a scene mask so that the actor builder builds the actor only in a subset of the environments
        """
        self.scene_mask = scene_mask

    def build_kinematic(self, name):
        self.set_physx_body_type("kinematic")
        return self.build(name=name)

    def build_static(self, name):
        self.set_physx_body_type("static")
        return self.build(name=name)

    def build(self, name):
        """
        Build the actor with the given name.

        Different to the original SAPIEN API, a unique name is required here.
        """
        self.set_name(name)

        num_actors = self.scene.num_envs
        if self.scene_mask is not None:
            assert (
                len(self.scene_mask) == self.scene.num_envs
            ), "Scene mask size is not correct. Must be the same as the number of sub scenes"
            num_actors = np.sum(num_actors)

        initial_pose = Pose.create(self.initial_pose)
        initial_pose_b = initial_pose.raw_pose.shape[0]
        assert initial_pose_b == 1 or initial_pose_b == num_actors
        initial_pose_np = to_numpy(initial_pose.raw_pose)

        entities = []
        parallelized = len(self.scene.sub_scenes) > 1

        i = 0
        for scene_idx, sub_scene in enumerate(self.scene.sub_scenes):
            if self.scene_mask is not None and self.scene_mask[i] == False:
                continue
            entity = self.build_entity()
            # prepend scene idx to entity name if there is more than one scene
            if parallelized:
                entity.name = f"scene-{scene_idx}_{self.name}"
            else:
                entity.name = self.name
            # set pose before adding to scene
            if initial_pose_b == 1:
                entity.pose = to_sapien_pose(initial_pose_np)
            else:
                entity.pose = to_sapien_pose(initial_pose_np[i])
            sub_scene.add_entity(entity)
            entities.append(entity)
            i += 1
        actor = Actor.create_from_entities(entities)

        # if it is a static body type and this is a GPU sim but we are given a single initial pose, we repeat it for the purposes of observations
        if (
            self.physx_body_type == "static"
            and initial_pose_b == 1
            and physx.is_gpu_enabled()
        ):
            actor._builder_initial_pose = Pose.create(
                initial_pose.raw_pose.repeat(num_actors, 1)
            )
        else:
            actor._builder_initial_pose = initial_pose
        self.scene.actors[self.name] = actor
        return actor
