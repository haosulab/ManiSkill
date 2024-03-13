"""
SceneBuilder for the AI2Thor scenes, using configurations and assets stored in https://huggingface.co/datasets/hssd/ai2thor-hab

"""


import json
import os.path as osp
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sapien
import sapien.core as sapien
import sapien.physx as physx
import transforms3d
from tqdm import tqdm

from mani_skill import ASSET_DIR
from mani_skill.agents.robots.fetch import FETCH_UNIQUE_COLLISION_BIT, Fetch
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.structs.actor import Actor

from .constants import SCENE_SOURCE_TO_DATASET, SceneConfig, load_ai2thor_metadata

DATASET_CONFIG_DIR = osp.join(osp.dirname(__file__), "metadata")

ALL_SCENE_CONFIGS = (
    dict()
)  # cached results mapping scene dataset ID to a list of scene configurations

OBJECT_SEMANTIC_ID_MAPPING, SEMANTIC_ID_OBJECT_MAPPING, MOVEABLE_OBJECT_IDS = (
    None,
    None,
    None,
)

# TODO (arth): fix coacd so this isn't necessary
WORKING_OBJS = ["apple", "potato", "tomato"]


class AI2THORBaseSceneBuilder(SceneBuilder):
    """
    The Base AI2THOR scene builder class. Subclasses
    """

    scene_dataset: str
    robot_init_qpos_noise: float = 0.02

    def __init__(self, env, robot_init_qpos_noise=0.02):
        self.env = env
        self.robot_init_qpos_noise = robot_init_qpos_noise
        global OBJECT_SEMANTIC_ID_MAPPING, SEMANTIC_ID_OBJECT_MAPPING, MOVEABLE_OBJECT_IDS
        (
            OBJECT_SEMANTIC_ID_MAPPING,
            SEMANTIC_ID_OBJECT_MAPPING,
            MOVEABLE_OBJECT_IDS,
        ) = load_ai2thor_metadata()
        self._scene_configs: List[SceneConfig] = []
        if self.scene_dataset not in ALL_SCENE_CONFIGS:
            dataset_path = SCENE_SOURCE_TO_DATASET[self.scene_dataset].metadata_path
            with open(osp.join(DATASET_CONFIG_DIR, dataset_path)) as f:
                scene_jsons = json.load(f)["scenes"]
            self._scene_configs += [
                SceneConfig(config_file=scene_json, source=self.scene_dataset)
                for scene_json in scene_jsons
            ]
            ALL_SCENE_CONFIGS[self.scene_dataset] = self._scene_configs
        else:
            self._scene_configs = ALL_SCENE_CONFIGS[self.scene_dataset]

        self._scene_navigable_positions = [None] * len(self._scene_configs)
        self.actor_default_poses: List[Tuple[Actor, sapien.Pose]] = []

        self.scene_idx = None

    def _should_be_kinematic(self, template_name: str):
        object_config_json = (
            Path(ASSET_DIR)
            / "scene_datasets/ai2thor/ai2thorhab-uncompressed/configs"
            / f"{template_name}.object_config.json"
        )
        with open(object_config_json, "r") as f:
            object_config_json = json.load(f)
        semantic_id = object_config_json["semantic_id"]
        object_category = SEMANTIC_ID_OBJECT_MAPPING[semantic_id]
        return object_category not in MOVEABLE_OBJECT_IDS

    # TODO (arth): figure out coacd building issues (currenlty fails on > 80% objs)
    def build(
        self, scene: ManiSkillScene, scene_idx=0, convex_decomposition="none", **kwargs
    ):
        # save scene and movable objects when building scene
        self._scene_objects: Dict[str, Actor] = dict()
        self._movable_objects: Dict[str, Actor] = dict()
        self.scene_idx = scene_idx

        scene_cfg = self._scene_configs[scene_idx]

        dataset = SCENE_SOURCE_TO_DATASET[scene_cfg.source]
        with open(osp.join(dataset.dataset_path, scene_cfg.config_file), "rb") as f:
            scene_json = json.load(f)

        bg_path = str(
            Path(ASSET_DIR)
            / "scene_datasets/ai2thor/ai2thor-hab/assets"
            / f"{scene_json['stage_instance']['template_name']}.glb"
        )
        builder = scene.create_actor_builder()

        bg_q = transforms3d.quaternions.axangle2quat(
            np.array([1, 0, 0]), theta=np.deg2rad(90)
        )
        if self.scene_dataset == "ProcTHOR":
            # for some reason the scene needs to rotate around y-axis by 90 degrees for ProcTHOR scenes from hssd dataset
            bg_q = transforms3d.quaternions.qmult(
                bg_q,
                transforms3d.quaternions.axangle2quat(
                    np.array([0, -1, 0]), theta=np.deg2rad(90)
                ),
            )
        bg_pose = sapien.Pose(q=bg_q)
        builder.add_visual_from_file(bg_path, pose=bg_pose)
        builder.add_nonconvex_collision_from_file(bg_path, pose=bg_pose)
        self.bg = builder.build_static(name="scene_background")

        global_id = 0
        for object in tqdm(scene_json["object_instances"][:]):
            model_path = (
                Path(ASSET_DIR)
                / "scene_datasets/ai2thor/ai2thorhab-uncompressed/assets"
                / f"{object['template_name']}.glb"
            )
            actor_id = f"{object['template_name']}_{global_id}"
            global_id += 1
            q = transforms3d.quaternions.axangle2quat(
                np.array([1, 0, 0]), theta=np.deg2rad(90)
            )
            rot_q = [
                object["rotation"][0],
                object["rotation"][1],
                object["rotation"][2],
                object["rotation"][-1],
            ]
            q = transforms3d.quaternions.qmult(q, rot_q)

            builder = scene.create_actor_builder()
            if self._should_be_kinematic(object["template_name"]) or not np.any(
                [name in actor_id.lower() for name in WORKING_OBJS]
            ):
                position = [
                    object["translation"][0],
                    -object["translation"][2],
                    object["translation"][1] + 0,
                ]
                position = [
                    object["translation"][0],
                    -object["translation"][2],
                    object["translation"][1] + 0,
                ]
                pose = sapien.Pose(p=position, q=q)
                builder.add_visual_from_file(str(model_path), pose=pose)
                builder.add_nonconvex_collision_from_file(str(model_path), pose=pose)
                actor = builder.build_static(name=actor_id)
            else:
                position = [
                    object["translation"][0],
                    -object["translation"][2],
                    object["translation"][1] + 0.005,
                ]
                builder.add_visual_from_file(str(model_path))
                builder.add_multiple_convex_collisions_from_file(
                    str(model_path),
                    decomposition=convex_decomposition,
                )
                actor = builder.build(name=actor_id)
                self._movable_objects[actor.name] = actor
                pose = sapien.Pose(p=position, q=q)
                self.actor_default_poses.append((actor, pose))
            self._scene_objects[actor_id] = actor

        if self._scene_navigable_positions[scene_idx] is None:
            self._scene_navigable_positions[scene_idx] = np.load(
                Path(dataset.dataset_path)
                / (
                    Path(scene_cfg.config_file).stem.split(".")[0]
                    + f".{self.env.robot_uids}.navigable_positions.npy"
                )
            )

    def disable_fetch_ground_collisions(self):
        for body in self.bg._bodies:
            cs = body.get_collision_shapes()[0]
            cg = cs.get_collision_groups()
            cg[2] |= FETCH_UNIQUE_COLLISION_BIT
            cs.set_collision_groups(cg)

    def set_actor_default_poses_vels(self):
        for actor, pose in self.actor_default_poses:
            actor.set_pose(pose)
            actor.set_linear_velocity([0, 0, 0])
            actor.set_angular_velocity([0, 0, 0])

    def initialize(self, env_idx):

        self.set_actor_default_poses_vels()

        if self.env.robot_uids == "panda":
            # fmt: off
            # EE at [0.615, 0, 0.17]
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            qpos[:-2] += self.env._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids == "xmate3_robotiq":
            qpos = np.array(
                [0, np.pi / 6, 0, np.pi / 3, 0, np.pi / 2, -np.pi / 2, 0, 0]
            )
            qpos[:-2] += self.env._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.562, 0, 0]))
        elif self.env.robot_uids == "fetch":
            agent: Fetch = self.env.agent
            agent.reset(agent.RESTING_QPOS)
            agent.robot.set_pose(sapien.Pose([*self.navigable_positions[0], 0.001]))
            self.disable_fetch_ground_collisions()
        else:
            raise NotImplementedError(self.env.robot_uids)

    @property
    def scene_configs(self) -> List[SceneConfig]:
        return self._scene_configs

    @property
    def navigable_positions(self) -> np.ndarray:
        assert isinstance(
            self.scene_idx, int
        ), "Must build scene before getting navigable positions"
        return self._scene_navigable_positions[self.scene_idx]

    @property
    def scene_objects(self) -> List[Actor]:
        return list(self._scene_objects.values())

    @property
    def movable_objects(self) -> List[Actor]:
        return list(self._movable_objects.values())

    @property
    def movable_objects_by_id(self) -> Dict[str, Actor]:
        return self._movable_objects
