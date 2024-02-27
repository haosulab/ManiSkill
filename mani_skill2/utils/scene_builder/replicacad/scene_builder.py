"""
Code for building scenes from the ReplicaCAD dataset https://aihabitat.org/datasets/replica_cad/

This code is also heavily commented to serve as a tutorial for how to build custom scenes from scratch and/or port scenes over from other datasets/simulators
"""

import json
import os.path as osp
from dataclasses import dataclass

import numpy as np
import sapien
import torch
import transforms3d

from mani_skill2 import ASSET_DIR
from mani_skill2.agents.robots.fetch.fetch import FETCH_UNIQUE_COLLISION_BIT, Fetch
from mani_skill2.envs.scene import ManiSkillScene
from mani_skill2.utils.scene_builder import SceneBuilder

DATASET_CONFIG_DIR = osp.join(osp.dirname(__file__), "metadata")


@dataclass
class SceneConfig:
    config_file: str
    # source: str
    # spawn_pos_file: str = None


class ReplicaCADSceneBuilder(SceneBuilder):
    def __init__(self, env, robot_init_qpos_noise=0.02):
        super().__init__(env, robot_init_qpos_noise)

        # Scene datasets from any source generally have several configurations, each of which may involve changing object geometries, poses etc.
        # You should store this configuration information in the self._scene_configs list, which permits the code to sample from when
        # simulating more than one scene or performing reconfiguration

        # for ReplicaCAD we have saved the list of all scene configurations from the dataset to a local json file and create SceneConfig objects out of it
        with open(osp.join(DATASET_CONFIG_DIR, "apts.json")) as f:
            self._scene_configs = json.load(f)["scenes"]

    def build(
        self, scene: ManiSkillScene, scene_idx=0, convex_decomposition="none", **kwargs
    ):
        """
        Given a ManiSkillScene, a sampled scene_idx, build/load the scene objects

        scene_idx is an index corresponding to a sampled scene config in self._scene_configs. The code should...
        TODO (stao): scene_idx should probably be replaced with scene config?
        """
        scene_cfg = self._scene_configs[scene_idx]

        # We read the json config file describing the scene setup for the selected ReplicaCAD scene
        with open(
            osp.join(
                ASSET_DIR,
                "scene_datasets/replica_cad_dataset/configs/scenes",
                scene_cfg,
            ),
            "rb",
        ) as f:
            scene_json = json.load(f)
        # import ipdb;ipdb.set_trace()
        background_template_name = osp.basename(
            scene_json["stage_instance"]["template_name"]
        )
        bg_path = f"data/scene_datasets/replica_cad_dataset/stages/{background_template_name}.glb"
        builder = scene.create_actor_builder()
        q = transforms3d.quaternions.axangle2quat(
            np.array([1, 0, 0]), theta=np.deg2rad(90)
        )
        bg_pose = sapien.Pose(q=q)
        builder.add_visual_from_file(bg_path, pose=bg_pose)

        builder.add_nonconvex_collision_from_file(bg_path, pose=bg_pose)
        self.bg = builder.build_static(name="scene_background")
        self.disable_fetch_ground_collisions()

        self.default_dynamic_actor_poses = []

        for obj in scene_json["object_instances"]:
            obj_path = osp.basename(obj["template_name"])
            obj_path = f"{obj_path}.object_config.json"
            obj_cfg_path = osp.join(
                ASSET_DIR,
                "scene_datasets/replica_cad_dataset/configs/objects",
                obj_path,
            )
            with open(obj_cfg_path) as f:
                obj_cfg = json.load(f)
            print(obj["template_name"], obj_cfg.keys())
            visual_file = osp.join(osp.dirname(obj_cfg_path), obj_cfg["render_asset"])
            if "collision_asset" in obj_cfg:
                collision_file = osp.join(
                    osp.dirname(obj_cfg_path), obj_cfg["collision_asset"]
                )
            builder = scene.create_actor_builder()
            pos = obj["translation"]
            rot = obj["rotation"]
            pose = sapien.Pose(q=q) * sapien.Pose(pos, rot)
            if obj["motion_type"] == "DYNAMIC":
                builder.add_visual_from_file(visual_file)
                if (
                    "use_bounding_box_for_collision" in obj_cfg
                    and obj_cfg["use_bounding_box_for_collision"]
                ):
                    builder.add_convex_collision_from_file(visual_file)
                else:
                    builder.add_multiple_convex_collisions_from_file(collision_file)
                actor = builder.build(name=obj_path)
                self.default_dynamic_actor_poses.append(
                    (actor, pose * sapien.Pose(p=[0, 0, 0.005]))
                )
            elif obj["motion_type"] == "STATIC":
                builder.add_visual_from_file(visual_file, pose=pose)
                builder.add_nonconvex_collision_from_file(visual_file, pose=pose)
                actor = builder.build_static(obj_path)

    def initialize(self, env_idx):
        if self.env.robot_uids == "fetch":
            agent: Fetch = self.env.agent
            agent.reset(agent.RESTING_QPOS)
            agent.robot.set_pose(sapien.Pose([-0.3, -0.5, 0.001]))

        else:
            raise NotImplementedError(self.env.robot_uids)
        for actor, pose in self.default_dynamic_actor_poses:
            actor.set_pose(pose)

    def disable_fetch_ground_collisions(self):
        # TODO (stao) (arth): is there a better way to model robots in sim. This feels very unintuitive.
        for body in self.bg._bodies:
            cs = body.get_collision_shapes()[0]
            cg = cs.get_collision_groups()
            cg[2] |= FETCH_UNIQUE_COLLISION_BIT
            cs.set_collision_groups(cg)
