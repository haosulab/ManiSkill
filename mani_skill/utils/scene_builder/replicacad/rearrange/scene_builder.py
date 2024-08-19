import json
import os
import os.path as osp
from typing import Dict, List
from pathlib import Path
from collections import defaultdict
import copy
import itertools
from functools import cached_property

import numpy as np
import sapien
import torch
import transforms3d

from mani_skill import ASSET_DIR
from mani_skill.utils.scene_builder.registration import register_scene_builder
from mani_skill.utils.structs import Actor
from mani_skill.utils.building import actors

from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder

HIDDEN_OBJ_COLLISION_GROUP = 30
HIDDEN_POSE = sapien.Pose(p=[99999] * 3)


@register_scene_builder("ReplicaCADRearrange")
class ReplicaCADRearrangeSceneBuilder(ReplicaCADSceneBuilder):

    task_names: List[str] = ["set_table:train"]

    # init configs for Rearrange stasks are default_object_poses for each object type
    init_configs: List[List[Dict[str, List[sapien.Pose]]]] = None

    def __init__(self, env):
        # the Habitat Rearrange tasks build on the base ReplicaCAD scenes by creating episode configs
        # each of which designate YCB object locations, articulation states, goal locations, etc
        # so, this scenebuilder will inherit from the ReplicaCADSceneBuilder
        #   1. the build_configs will be the base ReplicaCAD build configs
        #   2. we will load RCAD scenes into their own parallel env as directed by build(). we will furthermore
        #       split the Rearrange episode configs by which RCAD scene they use. each RCAD scene
        #       must have its own parallel env as they have some statically-built actors
        #   3. we then make as multiple copies of each YCB object in each parallel env. we also save
        #       we also save the pose data for each episode config
        #   4. finally, when initializing the scene, we use init_config_idxs to index the poses for that
        #       episode config, allowing us to initialize the scene
        # alternatively, we could rebuild each time we want to use a different episode config. this would be
        #   easier and allow for more "i.i.d." sampling of episode configs. however, using the above described
        #   method is much faster, as setting poses, qpos, etc is much faster than rebuilding/destroying actors

        # init base replicacad scene builder first
        super().__init__(env, include_staging_scenes=True)

        # load rearrange episode configs. we will not directly use these to build/init scenes,
        # but we will save certain data from these configs
        self._rearrange_configs = []
        for task_name in self.task_names:
            task, split = task_name.split(":")
            self._rearrange_configs += [
                osp.join(split, task, f)
                for f in sorted(
                    os.listdir(
                        osp.join(
                            ASSET_DIR,
                            "scene_datasets/replica_cad_dataset/rearrange",
                            split,
                            task,
                        ),
                    )
                )
                if f.endswith(".json")
            ]

        # keep track of which build config idxs are used for sampling
        self.used_build_config_idxs = set()
        bc_to_idx = dict((v, i) for i, v in enumerate(self.build_configs))
        for rc in self._rearrange_configs:
            with open(
                osp.join(
                    ASSET_DIR,
                    "scene_datasets/replica_cad_dataset/rearrange",
                    rc,
                ),
                "rb",
            ) as f:
                episode_json = json.load(f)
            self.used_build_config_idxs.add(
                bc_to_idx[Path(episode_json["scene_id"]).name]
            )

        self.before_hide_collision_groups: Dict[str, List[int]] = dict()

    def build(self, build_config_idxs: List[int]):
        assert all(
            [bci in self.used_build_config_idxs for bci in build_config_idxs]
        ), f"got one or more unused build_config_idxs in {build_config_idxs}; This RCAD Rearrange task only uses the following build_config_idxs: {self.used_build_config_idxs}"
        assert (
            len(build_config_idxs) == self.env.num_envs
        ), f"Got {len(build_config_idxs)} build_config_idxs but only have {self.env.num_envs} envs"

        # the build_config_idxs are idxs for the RCAD build configs
        # super().build builds the base RCAD scenes (including static objects)
        super().build(build_config_idxs)

        # EXTRACTING AND ORGANIZING EPISODE CONFIG DATA
        # here we extract the ycb object poses for each episode config and sort them by
        #   which base RCAD config they use
        # final init config list will map ith build config (i.e. the ith RCAD scene) to a list
        #   of default object poses for each episode config using the ith RCAD scene

        # get transformations to convert RCAD poses to poses that work in our scenes
        q = transforms3d.quaternions.axangle2quat(
            np.array([1, 0, 0]), theta=np.deg2rad(90)
        )
        world_transform = sapien.Pose(q=q).inv()
        obj_transform = sapien.Pose(q=q, p=[0, 0, 0.01])

        # self.rcad_to_rearrange_configs: which rearrange episode configs use each rcad config
        # default_object_poses: default poses for ycb objects from each rearrange episode config
        self.rcad_to_rearrange_configs: Dict[str, List[str]] = defaultdict(list)
        default_object_poses: Dict[str, Dict[str, List[sapien.Pose]]] = dict()
        for rc in self._rearrange_configs:
            objects: Dict[str, List[sapien.Pose]] = defaultdict(list)

            with open(
                osp.join(
                    ASSET_DIR,
                    "scene_datasets/replica_cad_dataset/rearrange",
                    rc,
                ),
                "rb",
            ) as f:
                episode_json = json.load(f)

            for actor_id, transformation in episode_json["rigid_objs"]:
                actor_id = actor_id.split(".")[0]
                objects[actor_id].append(
                    obj_transform * sapien.Pose(transformation) * world_transform
                )

            self.rcad_to_rearrange_configs[Path(episode_json["scene_id"]).name].append(
                rc
            )
            default_object_poses[rc] = objects

        # create init config
        self.init_configs: List[List[Dict[str, List[sapien.Pose]]]] = [
            [
                default_object_poses[rc]
                for rc in self.rcad_to_rearrange_configs[self.build_configs[bci]]
            ]
            for bci in build_config_idxs
        ]

        # YCB OBJ BUILDING
        # instead of making new YCB objs for each episode config, we will
        # build multiple instances of each YCB object for each RCAD scene
        # when initializing, we set poses according to the sampled init configs,
        # and we hide any unused instances of YCB objects
        rcad_config_to_num_ycb_objs_to_build: Dict[str, Dict[str, int]] = dict()
        for rcad_config in self.rcad_to_rearrange_configs.keys():
            obj_ids = set()
            for rearrange_config in self.rcad_to_rearrange_configs[rcad_config]:
                for obj_id in default_object_poses[rearrange_config].keys():
                    obj_ids.add(obj_id)

            num_ycb_objs_to_build = dict()
            for obj_id in obj_ids:
                num_ycb_objs_to_build[obj_id] = max(
                    [
                        len(default_object_poses[rearrange_config][obj_id])
                        for rearrange_config in self.rcad_to_rearrange_configs[
                            rcad_config
                        ]
                    ]
                )

            rcad_config_to_num_ycb_objs_to_build[rcad_config] = num_ycb_objs_to_build

        # save num init configs per build config for init config sampling
        self.num_init_configs_per_build_config = [
            len(self.rcad_to_rearrange_configs[self.build_configs[bci]])
            for bci in build_config_idxs
        ]

        # find max number of each ycb obj needed to support all init configs in each parallel env
        self.ycb_objs_per_env = []
        for env_num, bci in enumerate(build_config_idxs):
            rcad_config = self.build_configs[bci]
            num_ycb_objs_to_build = rcad_config_to_num_ycb_objs_to_build[rcad_config]

            ycb_objs = defaultdict(list)
            for actor_id, num_objs in num_ycb_objs_to_build.items():
                for no in range(num_objs):
                    obj_instance_name = f"env-{env_num}_{actor_id}-{no}"
                    builder = actors.get_actor_builder(self.scene, id=f"ycb:{actor_id}")
                    builder.set_scene_idxs([env_num])
                    actor = builder.build(name=obj_instance_name)

                    ycb_objs[actor_id].append(actor)
                    self.scene_objects[obj_instance_name] = actor
                    self.movable_objects[obj_instance_name] = actor
            self.ycb_objs_per_env.append(ycb_objs)

    def initialize(self, env_idx: torch.Tensor, init_config_idxs: List[int]):
        assert all(
            [isinstance(bci, int) for bci in init_config_idxs]
        ), f"init_config_idxs should be list of ints, instead got {init_config_idxs}"
        assert len(init_config_idxs) == self.env.num_envs

        # initialize base scenes
        super().initialize(env_idx)

        # get sampled init configs
        sampled_init_configs = [
            env_init_configs[idx]
            for env_init_configs, idx in zip(self.init_configs, init_config_idxs)
        ]

        # if pose given by init config, set pose
        # if pose not given, hide extra ycb obj by teleporting away and
        #       turning off collisions with other hidden objs
        for ycb_objs, init_poses in zip(self.ycb_objs_per_env, sampled_init_configs):
            for obj_name in ycb_objs:
                for obj, pose in itertools.zip_longest(
                    ycb_objs[obj_name], init_poses[obj_name], fillvalue=None
                ):
                    if pose is None:
                        self.hide_actor(obj)
                    else:
                        self.show_actor(obj, pose)

    def sample_build_config_idxs(self):
        used_build_config_idxs = list(self.used_build_config_idxs)
        return [
            used_build_config_idxs[i]
            for i in torch.randint(
                low=0, high=len(used_build_config_idxs), size=(self.env.num_envs,)
            ).tolist()
        ]

    def sample_init_config_idxs(self):
        low = torch.zeros(self.env.num_envs)
        high = torch.tensor(self.num_init_configs_per_build_config)
        size = (self.env.num_envs,)
        return (torch.randint(2**63 - 1, size=size) % (high - low) + low).int().tolist()

    def hide_actor(self, actor: Actor):
        actor.set_collision_group_bit(
            group=2, bit_idx=HIDDEN_OBJ_COLLISION_GROUP, bit=1
        )
        actor.set_pose(HIDDEN_POSE)

    def show_actor(self, actor: Actor, pose: sapien.Pose):
        actor.set_collision_group_bit(
            group=2, bit_idx=HIDDEN_OBJ_COLLISION_GROUP, bit=0
        )
        actor.set_pose(pose)

    @cached_property
    def init_config_names_to_idxs(self) -> int:

        _init_config_names_to_idx = dict()
        for rcad_config, rearrange_configs in self.rcad_to_rearrange_configs.items():
            for i, rc in enumerate(rearrange_configs):
                _init_config_names_to_idx[rc] = i

        return _init_config_names_to_idx
