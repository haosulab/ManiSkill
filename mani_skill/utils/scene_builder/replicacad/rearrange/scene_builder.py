import copy
import itertools
import json
import os
import os.path as osp
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sapien
import sapien.physx as physx
import torch
import transforms3d

from mani_skill import ASSET_DIR
from mani_skill.utils import common
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder
from mani_skill.utils.structs import Actor, Articulation

DEFAULT_HIDDEN_POS = [-10_000] * 3


class ReplicaCADRearrangeSceneBuilder(ReplicaCADSceneBuilder):

    task_names: list[str] = ["set_table:train"]

    # init configs for Rearrange stasks are default_object_poses for each object type
    init_configs: list[list[dict[str, list[sapien.Pose]]]] = None

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

    def build(
        self, build_config_idxs: list[int], init_config_names: Optional[list] = None
    ):
        if isinstance(build_config_idxs, int):
            build_config_idxs = [build_config_idxs] * self.env.num_envs
        assert all(
            [bci in self.used_build_config_idxs for bci in build_config_idxs]
        ), f"got one or more unused build_config_idxs in {build_config_idxs}; This RCAD Rearrange task only uses the following build_config_idxs: {self.used_build_config_idxs}"
        assert (
            len(build_config_idxs) == self.env.num_envs
        ), f"Got {len(build_config_idxs)} build_config_idxs but only have {self.env.num_envs} envs"

        # delete cached properties which are dependent on values recomputed at build time
        self.__dict__.pop("init_config_names_to_idxs", None)

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
        self.rcad_to_rearrange_configs: dict[str, list[str]] = dict()
        default_object_poses: dict[str, dict[str, list[sapien.Pose]]] = dict()
        if init_config_names is None:
            init_config_names = self._rearrange_configs
        for rc in init_config_names:
            objects: dict[str, list[sapien.Pose]] = defaultdict(list)

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

            rcad_config_name = Path(episode_json["scene_id"]).name
            if rcad_config_name not in self.rcad_to_rearrange_configs:
                self.rcad_to_rearrange_configs[rcad_config_name] = []
            self.rcad_to_rearrange_configs[rcad_config_name].append(rc)
            default_object_poses[rc] = objects

        # create init config
        self.init_configs: list[list[dict[str, list[sapien.Pose]]]] = [
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
        rcad_config_to_num_ycb_objs_to_build: dict[str, dict[str, int]] = dict()
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
        _running_default_hidden_pos_by_env = dict(
            (env_num, copy.deepcopy(DEFAULT_HIDDEN_POS))
            for env_num in range(len(build_config_idxs))
        )
        self._default_hidden_poses: dict[Actor, sapien.Pose] = dict()
        for env_num, bci in enumerate(build_config_idxs):
            rcad_config = self.build_configs[bci]
            num_ycb_objs_to_build = rcad_config_to_num_ycb_objs_to_build[rcad_config]

            ycb_objs = defaultdict(list)
            for actor_id, num_objs in num_ycb_objs_to_build.items():
                for no in range(num_objs):
                    obj_instance_name = f"env-{env_num}_{actor_id}-{no}"
                    builder = actors.get_actor_builder(self.scene, id=f"ycb:{actor_id}")
                    builder.set_scene_idxs([env_num])
                    builder.initial_pose = sapien.Pose(
                        p=_running_default_hidden_pos_by_env[env_num]
                    )
                    actor = builder.build(name=obj_instance_name)

                    ycb_objs[actor_id].append(actor)
                    self._default_hidden_poses[actor] = sapien.Pose(
                        p=_running_default_hidden_pos_by_env[env_num]
                    )
                    _running_default_hidden_pos_by_env[env_num][2] -= 10
                    self.scene_objects[obj_instance_name] = actor
                    self.movable_objects[obj_instance_name] = actor
            self.ycb_objs_per_env.append(ycb_objs)

        self.rcad_config_to_rearrange_ao_states: dict[
            str, list[list[Tuple[Articulation, torch.Tensor]]]
        ] = defaultdict(list)
        for env_num, bci in enumerate(build_config_idxs):
            rcad_config = self.build_configs[bci]
            rearrange_ao_states: list[list[Tuple[Articulation, torch.Tensor]]] = []
            for rearrange_config in self.rcad_to_rearrange_configs[rcad_config]:
                with open(
                    osp.join(
                        ASSET_DIR,
                        "scene_datasets/replica_cad_dataset/rearrange",
                        rearrange_config,
                    ),
                    "rb",
                ) as f:
                    episode_json = json.load(f)
                episode_ao_states: dict[str, dict[str, int]] = episode_json["ao_states"]
                articulation_qpos_pairs: list[Tuple[Articulation, torch.Tensor]] = []
                for base_articulation_id, qpos_dict in episode_ao_states.items():
                    aid, anum = base_articulation_id.split(":")
                    articulation = self.articulations[
                        f"env-{env_num}_{aid[:-1]}-{int(anum)}"
                    ]
                    base_qpos = torch.zeros(
                        articulation.max_dof, device=self.env.device
                    )
                    for link_num, qpos_val in qpos_dict.items():
                        qpos_idx = articulation.active_joints.index(
                            articulation.joints[int(link_num)]
                        )
                        base_qpos[qpos_idx] = qpos_val
                    articulation_qpos_pairs.append((articulation, base_qpos))
                rearrange_ao_states.append(articulation_qpos_pairs)
            self.rcad_config_to_rearrange_ao_states[rcad_config] = rearrange_ao_states

    # TODO (arth): fix this to work with partial resets
    def initialize(self, env_idx: torch.Tensor, init_config_idxs: list[int]):
        assert all(
            [isinstance(ici, int) for ici in init_config_idxs]
        ), f"init_config_idxs should be list of ints, instead got {init_config_idxs}"

        init_config_idxs: torch.Tensor = common.to_tensor(
            init_config_idxs, device=self.env.device
        ).to(torch.int)
        if env_idx.numel() != init_config_idxs.numel():
            init_config_idxs = init_config_idxs[env_idx]

        # get sampled init configs
        sampled_init_configs = [
            self.init_configs[env_num][idx]
            for env_num, idx in zip(env_idx, init_config_idxs)
        ]

        # sometimes, poses from end of prev episode can cause issues with articulations
        # we can avoid this by setting poses, but shifting non-hidden objects away from the scene
        for env_num, init_poses in zip(env_idx, sampled_init_configs):
            ycb_objs = self.ycb_objs_per_env[env_num]
            for obj_name in ycb_objs:
                for obj, pose in itertools.zip_longest(
                    ycb_objs[obj_name], init_poses[obj_name], fillvalue=None
                ):
                    if pose is None:
                        self.hide_actor(obj)
                    else:
                        temp_p = pose.p
                        temp_p[..., 2] += 1000
                        self.show_actor(obj, sapien.Pose(q=pose.q, p=temp_p))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene._gpu_fetch_all()

        # initialize base scenes
        super().initialize(env_idx)

        # teleport robot away for init
        self.env.agent.robot.set_pose(sapien.Pose([-10, 0, -100]))

        # if pose given by init config, set pose
        # if pose not given, hide extra ycb obj by teleporting away and
        #       turning off collisions with other hidden objs
        for env_num, init_poses in zip(env_idx, sampled_init_configs):
            ycb_objs = self.ycb_objs_per_env[env_num]
            for obj_name in ycb_objs:
                for obj, pose in itertools.zip_longest(
                    ycb_objs[obj_name], init_poses[obj_name], fillvalue=None
                ):
                    if pose is None:
                        self.hide_actor(obj)
                    else:
                        self.show_actor(obj, pose)

        # set articulation qpos as needed
        # TODO (arth): also shift objs inside the articulation as needed
        # NOTE (arth): for now the rearrange configs only have the fridge move,
        #       so above only an issue when one can generate their own
        #       set the kitchen counter in the ao_configs
        for env_num, ici in zip(env_idx, init_config_idxs):
            rcad_config = self.build_configs[self.build_config_idxs[env_num]]
            ao_states = self.rcad_config_to_rearrange_ao_states[rcad_config][ici]
            for articulation, qpos in ao_states:
                articulation_scene_idxs = articulation._scene_idxs.tolist()
                base_qpos = articulation.qpos
                base_qpos[articulation_scene_idxs.index(env_num)] *= 0
                base_qpos[articulation_scene_idxs.index(env_num)] = qpos
                reset_idxs = [
                    bn
                    for bn, en in enumerate(articulation._scene_idxs)
                    if en in env_idx
                ]
                articulation.set_qpos(base_qpos[reset_idxs])
                articulation.set_qvel(articulation.qvel[reset_idxs] * 0)

        if self.scene.gpu_sim_enabled and len(env_idx) == self.env.num_envs:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene.px.step()
            self.scene._gpu_fetch_all()

        # teleport robot back to correct location
        if self.env.robot_uids == "fetch":
            self.env.agent.reset(self.env.agent.keyframes["rest"].qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-1, 0, 0.02]))
        else:
            raise NotImplementedError(self.env.robot_uids)

    def sample_build_config_idxs(self):
        used_build_config_idxs = list(self.used_build_config_idxs)
        return [
            used_build_config_idxs[i]
            for i in torch.randint(
                low=0, high=len(used_build_config_idxs), size=(self.env.num_envs,)
            ).tolist()
        ]

    def sample_init_config_idxs(self):
        low = torch.zeros(self.env.num_envs, dtype=torch.int)
        high = torch.tensor(self.num_init_configs_per_build_config).int()
        size = (self.env.num_envs,)
        return (
            (torch.randint(2**63 - 1, size=size) % (high - low).int() + low)
            .int()
            .tolist()
        )

    def hide_actor(self, actor: Actor):
        actor.set_pose(self._default_hidden_poses[actor])

    def show_actor(self, actor: Actor, pose: sapien.Pose):
        actor.set_pose(pose)

    @cached_property
    def init_config_names_to_idxs(self) -> int:

        _init_config_names_to_idx = dict()
        for rearrange_configs in self.rcad_to_rearrange_configs.values():
            for i, rc in enumerate(rearrange_configs):
                _init_config_names_to_idx[rc] = i

        return _init_config_names_to_idx
