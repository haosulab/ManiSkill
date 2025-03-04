"""
Code for building scenes from the ReplicaCAD dataset https://aihabitat.org/datasets/replica_cad/

This code is also heavily commented to serve as a tutorial for how to build custom scenes from scratch and/or port scenes over from other datasets/simulators
"""

import json
import os.path as osp
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch
import transforms3d
import trimesh

from mani_skill import ASSET_DIR
from mani_skill.agents.robots.fetch import (
    FETCH_BASE_COLLISION_BIT,
    FETCH_WHEELS_COLLISION_BIT,
)
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.scene_builder.registration import register_scene_builder
from mani_skill.utils.structs import Actor, Articulation

DATASET_CONFIG_DIR = osp.join(osp.dirname(__file__), "metadata")

IGNORE_FETCH_COLLISION_STRS = ["mat", "rug", "carpet"]


@register_scene_builder("ReplicaCAD")
class ReplicaCADSceneBuilder(SceneBuilder):

    robot_initial_pose = sapien.Pose(
        p=[-1, 0, 0.02]
    )  # generally a safe initial spawn pose for the Fetch robot

    builds_lighting = True  # we set this true because the ReplicaCAD dataset defines some lighting for us so we don't need the default option from ManiSkill

    # build configs for RCAD are string file names
    build_configs: List[str] = None

    def __init__(self, env, robot_init_qpos_noise=0.02, include_staging_scenes=False):
        super().__init__(env, robot_init_qpos_noise=robot_init_qpos_noise)
        # Scene datasets from any source generally have several configurations, each of which may involve changing object geometries, poses etc.
        # You should store this configuration information in the self.build_configs list, which permits the code to sample from when
        # simulating more than one scene or performing reconfiguration

        # for ReplicaCAD we have saved the list of all scene configuration files from the dataset to a local json file
        with open(osp.join(DATASET_CONFIG_DIR, "scene_configs.json")) as f:
            build_config_json = json.load(f)
            self.build_configs = build_config_json["scenes"]
            if include_staging_scenes:
                self.build_configs += build_config_json["staging_scenes"]

        # cache navigable positions from files
        # assumes navigable position files saved
        self._navigable_positions = [None] * len(self.build_configs)
        self.build_config_idxs: List[int] = None

    def build(self, build_config_idxs: Union[int, List[int]]):
        # build_config_idxs is a list of integers, where the ith value is the scene idx for the ith parallel env
        if isinstance(build_config_idxs, int):
            build_config_idxs = [build_config_idxs] * self.env.num_envs
        assert all([isinstance(bci, int) for bci in build_config_idxs])
        assert len(build_config_idxs) == self.env.num_envs

        # delete cached properties which are dependent on values recomputed at build time
        self.__dict__.pop("ray_traced_lighting", None)

        # Keep track of movable and static objects, build_config_idxs for envs, and poses
        self.build_config_idxs = build_config_idxs
        self.scene_objects: Dict[str, Actor] = dict()
        self.movable_objects: Dict[str, Actor] = dict()
        self.articulations: Dict[str, Articulation] = dict()
        self._default_object_poses: List[Tuple[Actor, sapien.Pose]] = []

        # keep track of background objects separately as we need to disable mobile robot collisions
        # note that we will create a merged actor using these objects to represent the bg
        bgs = [None] * self.env.num_envs
        for bci in np.unique(build_config_idxs):
            """
            Given a list of sampled build_config_idxs, build/load the scene objects

            TODO (arth): return builder option to set static/dynamic in env
                -  for now leave as-is since has smaller change in performance
            """

            env_idx = [i for i, v in enumerate(build_config_idxs) if v == bci]
            unique_id = "scs-" + str(env_idx).replace(" ", "")
            build_config_path = self.build_configs[bci]

            # We read the json config file describing the scene setup for the selected ReplicaCAD scene
            with open(
                osp.join(
                    ASSET_DIR,
                    "scene_datasets/replica_cad_dataset/configs/scenes",
                    build_config_path,
                ),
                "rb",
            ) as f:
                build_config_json = json.load(f)

            # The complex part of porting over scene datasets is that each scene dataset often has it's own format and there is no
            # one size fits all solution to read that format and use it. The best way to port a scene dataset over is to look
            # at the configuration files, get a sense of the pattern and find how they reference .glb model files and potentially
            # decomposed convex meshes for physical simulation

            # ReplicaCAD stores the background model here
            background_template_name = osp.basename(
                build_config_json["stage_instance"]["template_name"]
            )
            bg_path = str(
                ASSET_DIR
                / f"scene_datasets/replica_cad_dataset/stages/{background_template_name}.glb"
            )
            builder = self.scene.create_actor_builder()
            # Note all ReplicaCAD assets are rotated by 90 degrees as they use a different xyz convention to SAPIEN/ManiSkill.
            q = transforms3d.quaternions.axangle2quat(
                np.array([1, 0, 0]), theta=np.deg2rad(90)
            )
            bg_pose = sapien.Pose(q=q)

            # When creating objects that do not need to be moved ever, you must provide the pose of the object ahead of time
            # and use builder.build_static. Objects such as the scene background (also called a stage) fits in this category
            builder.add_visual_from_file(bg_path)
            builder.add_nonconvex_collision_from_file(bg_path)
            builder.initial_pose = bg_pose
            builder.set_scene_idxs(env_idx)
            bg = builder.build_static(name=f"{unique_id}_scene_background")
            for i, env_num in enumerate(env_idx):
                bgs[env_num] = bg._objs[i]

            # In scenes, there will always be dynamic objects, kinematic objects, and static objects.
            # In the case of ReplicaCAD there are only dynamic and static objects. Since dynamic objects can be moved during simulation
            # we need to keep track of the initial poses of each dynamic actor we create.
            for obj_num, obj_meta in enumerate(build_config_json["object_instances"]):

                # Again, for any dataset you will have to figure out how they reference object files
                # Note that ASSET_DIR will always refer to the ~/.maniskill/data folder or whatever MS_ASSET_DIR is set to
                obj_config_path = osp.join(
                    ASSET_DIR,
                    "scene_datasets/replica_cad_dataset/configs/objects",
                    f"{osp.basename(obj_meta['template_name'])}.object_config.json",
                )
                with open(obj_config_path) as f:
                    obj_config = json.load(f)
                visual_file = osp.join(
                    osp.dirname(obj_config_path), obj_config["render_asset"]
                )
                if "collision_asset" in obj_config:
                    collision_file = osp.join(
                        osp.dirname(obj_config_path), obj_config["collision_asset"]
                    )
                builder = self.scene.create_actor_builder()
                pos = obj_meta["translation"]
                rot = obj_meta["rotation"]
                # left multiplying by the offset quaternion we used for the stage/scene background as all assets in ReplicaCAD are rotated by 90 degrees
                pose = sapien.Pose(q=q) * sapien.Pose(pos, rot)

                actor_name = f'{obj_meta["template_name"]}-{obj_num}'
                # Neatly for simulation, ReplicaCAD specifies if an object is meant to be simulated as dynamic (can be moved like pots) or static (must stay still, like kitchen counters)
                if obj_meta["motion_type"] == "DYNAMIC":
                    builder.add_visual_from_file(visual_file)
                    if (
                        "use_bounding_box_for_collision" in obj_config
                        and obj_config["use_bounding_box_for_collision"]
                    ):
                        # some dynamic objects do not have decomposed convex meshes and instead should use a simple bounding box for collision detection
                        # in this case we use the add_convex_collision_from_file function of SAPIEN which just creates a convex collision based on the visual mesh
                        builder.add_convex_collision_from_file(visual_file)
                    else:
                        builder.add_multiple_convex_collisions_from_file(collision_file)
                    builder.initial_pose = pose
                    builder.set_scene_idxs(env_idx)
                    actor = builder.build(name=f"{unique_id}_{actor_name}")
                    self._default_object_poses.append((actor, pose))

                    # Add dynamic objects to movable_objects
                    for env_num in env_idx:
                        self.movable_objects[f"env-{env_num}_{actor_name}"] = actor
                elif obj_meta["motion_type"] == "STATIC":
                    builder.add_visual_from_file(visual_file)
                    # for static (and dynamic) objects you don't need to use pre convex decomposed meshes and instead can directly
                    # add the non convex collision mesh based on the visual mesh
                    builder.add_nonconvex_collision_from_file(visual_file)
                    builder.initial_pose = pose
                    builder.set_scene_idxs(env_idx)
                    actor = builder.build_static(name=f"{unique_id}_{actor_name}")

                # Add all objects to scene_objects
                for env_num in env_idx:
                    self.scene_objects[f"env-{env_num}_{actor_name}"] = actor

                # Certain objects, such as mats, rugs, and carpets, are on the ground and should not collide with the Fetch base
                if any([x in actor_name for x in IGNORE_FETCH_COLLISION_STRS]):
                    self.disable_fetch_move_collisions(
                        actor, disable_base_collisions=True
                    )

            # ReplicaCAD also provides articulated objects
            articulation_to_num = defaultdict(int)
            for i, articulated_meta in enumerate(
                build_config_json["articulated_object_instances"]
            ):

                template_name = articulated_meta["template_name"]
                if "door" in template_name:
                    continue
                pos = articulated_meta["translation"]
                rot = articulated_meta["rotation"]
                urdf_path = osp.join(
                    ASSET_DIR,
                    f"scene_datasets/replica_cad_dataset/urdf/{template_name}/{template_name}.urdf",
                )
                urdf_loader = self.scene.create_urdf_loader()
                articulation_name = (
                    f"{template_name}-{articulation_to_num[template_name]}"
                )
                urdf_loader.name = f"{unique_id}_{articulation_name}"
                urdf_loader.fix_root_link = articulated_meta["fixed_base"]
                urdf_loader.disable_self_collisions = True
                if "uniform_scale" in articulated_meta:
                    urdf_loader.scale = articulated_meta["uniform_scale"]
                builder = urdf_loader.parse(urdf_path)["articulation_builders"][0]
                pose = sapien.Pose(q=q) * sapien.Pose(pos, rot)
                builder.initial_pose = pose
                builder.set_scene_idxs(env_idx)
                articulation = builder.build()
                self._default_object_poses.append((articulation, pose))

                # for now classify articulated objects as "movable" object
                for env_num in env_idx:
                    self.articulations[f"env-{env_num}_{articulation_name}"] = (
                        articulation
                    )
                    self.scene_objects[f"env-{env_num}_{articulation_name}"] = (
                        articulation
                    )

                for link in articulation.links:
                    link.set_collision_group_bit(
                        group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
                    )

                articulation_to_num[template_name] += 1

            if self._navigable_positions[bci] is None:
                mesh_fp = (
                    Path(ASSET_DIR)
                    / "scene_datasets/replica_cad_dataset/configs/scenes"
                    / (
                        Path(self.build_configs[bci]).stem
                        + f".{str(self.env.robot_uids)}.navigable_positions.obj"
                    )
                )
                if mesh_fp.exists():
                    self._navigable_positions[bci] = trimesh.load(mesh_fp)

        # # ReplicaCAD's lighting isn't great for raytracing, so we define our own
        if self.ray_traced_lighting:
            for sub_scene in self.scene.sub_scenes:
                sub_scene.set_environment_map(
                    str(
                        (
                            Path(__file__).parent / "autumn_field_puresky_4k.hdr"
                        ).absolute()
                    )
                )
        else:
            self.scene.set_ambient_light([0.3] * 3)
        color = np.array([1.0, 0.8, 0.5]) * 2
        # entrance
        self.scene.add_point_light([-1.1, 2.775, 2.3], color=color)
        # dining area
        self.scene.add_point_light([-0.5, -1.44, 2.3], color=color)
        # dining back
        self.scene.add_point_light([2.4, -1.6, 2.3], color=color)
        # living room
        self.scene.add_point_light([2.5, -6.1, 2.3], color=color)
        # stair
        self.scene.add_point_light([3.14, 3.24, 3], color=color)

        # merge actors into one
        self.bg = Actor.create_from_entities(
            bgs,
            scene=self.scene,
            scene_idxs=torch.arange(self.env.num_envs, dtype=int),
            shared_name="scene_background",
        )

        # For the purposes of physical simulation, we disable collisions between the Fetch robot and the scene background
        self.disable_fetch_move_collisions(self.bg)

    def initialize(self, env_idx: torch.Tensor):

        # teleport robot away for init
        self.env.agent.robot.set_pose(sapien.Pose([-10, 0, -100]))

        for obj, pose in self._default_object_poses:
            obj.set_pose(pose)
            if isinstance(obj, Articulation):
                # note that during initialization you may only ever change poses/qpos of objects in scenes being reset
                obj.set_qpos(obj.qpos[0] * 0)
                obj.set_qvel(obj.qvel[0] * 0)

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

    def disable_fetch_move_collisions(
        self,
        actor: Actor,
        disable_base_collisions=False,
    ):
        actor.set_collision_group_bit(
            group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
        )
        if disable_base_collisions:
            actor.set_collision_group_bit(
                group=2, bit_idx=FETCH_BASE_COLLISION_BIT, bit=1
            )

    @property
    def navigable_positions(self) -> List[trimesh.Trimesh]:
        return [self._navigable_positions[bci] for bci in self.build_config_idxs]

    @cached_property
    def ray_traced_lighting(self) -> bool:
        return self.env._custom_human_render_camera_configs.get(
            "shader_pack", None
        ) in [
            "rt",
            "rt-fast",
        ]
