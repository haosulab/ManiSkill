"""
SceneBuilder for the AI2Thor scenes, using configurations and assets stored in https://huggingface.co/datasets/hssd/ai2thor-hab

"""

import json
import os.path as osp
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import sapien
import sapien.core as sapien
import sapien.physx as physx
import torch
import transforms3d
from tqdm import tqdm

from mani_skill import ASSET_DIR
from mani_skill.agents.robots.fetch import (
    FETCH_BASE_COLLISION_BIT,
    FETCH_WHEELS_COLLISION_BIT,
    Fetch,
)
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.structs import Actor, Articulation, Pose

from .constants import SCENE_SOURCE_TO_DATASET, AI2BuildConfig, load_ai2thor_metadata

DATASET_CONFIG_DIR = osp.join(osp.dirname(__file__), "metadata")

ALL_SCENE_CONFIGS = (
    dict()
)  # cached results mapping scene dataset ID to a list of scene configurations

OBJECT_SEMANTIC_ID_MAPPING, SEMANTIC_ID_OBJECT_MAPPING, MOVEABLE_OBJECT_IDS = (
    None,
    None,
    None,
)

WORKING_OBJS = [
    "apple",
    "potato",
    "tomato",
    "lettuce",
    "soap",
    "sponge",
    "plate",
    "book",
]
FETCH_BUILD_CONFIG_IDX_TO_START_POS = {
    0: (-3, 0),
    1: (-2, -2),
    2: (0, 0),
    3: (-3.5, 0),
    4: (0, -2),
    5: (-1, 1.5),
    6: (1, -0.5),
    7: (3.25, 1),
    8: (1, 2),
    9: (1, 1),
}


class AI2THORBaseSceneBuilder(SceneBuilder):
    """
    The Base AI2THOR scene builder class. Subclasses
    """

    scene_dataset: str

    def __init__(self, env, robot_init_qpos_noise=0.02):
        super().__init__(env, robot_init_qpos_noise=robot_init_qpos_noise)
        global OBJECT_SEMANTIC_ID_MAPPING, SEMANTIC_ID_OBJECT_MAPPING, MOVEABLE_OBJECT_IDS
        (
            OBJECT_SEMANTIC_ID_MAPPING,
            SEMANTIC_ID_OBJECT_MAPPING,
            MOVEABLE_OBJECT_IDS,
        ) = load_ai2thor_metadata()
        self.build_configs: List[AI2BuildConfig] = []
        if self.scene_dataset not in ALL_SCENE_CONFIGS:
            dataset_path = SCENE_SOURCE_TO_DATASET[self.scene_dataset].metadata_path
            with open(osp.join(DATASET_CONFIG_DIR, dataset_path)) as f:
                build_config_jsons = json.load(f)["scenes"]
            self.build_configs += [
                AI2BuildConfig(config_file=build_config_json, source=self.scene_dataset)
                for build_config_json in build_config_jsons
            ]
            ALL_SCENE_CONFIGS[self.scene_dataset] = self.build_configs
        else:
            self.build_configs = ALL_SCENE_CONFIGS[self.scene_dataset]

        self._navigable_positions = [None] * len(self.build_configs)
        self.build_config_idxs: List[int] = None

    def _should_be_static(self, template_name: str):
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
        self,
        build_config_idxs: Union[int, List[int]],
        convex_decomposition="none",
    ):
        # build_config_idxs is a list of integers, where the ith value is the scene idx for the ith parallel env
        if isinstance(build_config_idxs, int):
            build_config_idxs = [build_config_idxs] * self.env.num_envs
        assert all([isinstance(bci, int) for bci in build_config_idxs])
        assert len(build_config_idxs) == self.env.num_envs

        # save scene and movable objects when building scene
        self.build_config_idxs = build_config_idxs
        self.scene_objects: Dict[str, Actor] = dict()
        self.movable_objects: Dict[str, Actor] = dict()
        self.articulations: Dict[str, Articulation] = dict()
        self._default_object_poses: List[Tuple[Actor, sapien.Pose]] = []

        # keep track of background objects separately as we need to disable mobile robot collisions
        # note that we will create a merged actor using these objects to represent the bg
        bgs = [None] * self.env.num_envs

        for bci in np.unique(build_config_idxs):

            env_idx = [i for i, v in enumerate(build_config_idxs) if v == bci]
            unique_id = "scs-" + str(env_idx).replace(" ", "")
            build_config: AI2BuildConfig = self.build_configs[bci]

            dataset = SCENE_SOURCE_TO_DATASET[build_config.source]
            with open(
                osp.join(dataset.dataset_path, build_config.config_file), "rb"
            ) as f:
                build_config_json = json.load(f)

            bg_path = str(
                Path(ASSET_DIR)
                / "scene_datasets/ai2thor/ai2thor-hab/assets"
                / f"{build_config_json['stage_instance']['template_name']}.glb"
            )
            builder = self.scene.create_actor_builder()

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
            builder.set_scene_idxs(env_idx)
            bg = builder.build_static(name=f"{unique_id}_scene_background")
            for i, env_num in enumerate(env_idx):
                bgs[env_num] = bg._objs[i]

            global_id = 0
            for object in tqdm(build_config_json["object_instances"][:]):
                model_path = (
                    Path(ASSET_DIR)
                    / "scene_datasets/ai2thor/ai2thorhab-uncompressed/assets"
                    / f"{object['template_name']}.glb"
                )
                actor_name = f"{object['template_name']}_{global_id}"
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

                builder = self.scene.create_actor_builder()
                if self._should_be_static(object["template_name"]) or not any(
                    [name in actor_name.lower() for name in WORKING_OBJS]
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
                    builder.add_nonconvex_collision_from_file(
                        str(model_path), pose=pose
                    )
                    builder.set_scene_idxs(env_idx)
                    actor = builder.build_static(name=f"{unique_id}_{actor_name}")
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
                    builder.set_scene_idxs(env_idx)
                    actor = builder.build(name=f"{unique_id}_{actor_name}")

                    pose = sapien.Pose(p=position, q=q)
                    self._default_object_poses.append((actor, pose))

                    for env_num in env_idx:
                        self.movable_objects[f"env-{env_num}_{actor_name}"] = actor

                for env_num in env_idx:
                    self.scene_objects[f"env-{env_num}_{actor_name}"] = actor

            if self._navigable_positions[bci] is None:
                npy_fp = Path(dataset.dataset_path) / (
                    Path(build_config.config_file).stem.split(".")[0]
                    + f".{self.env.robot_uids}.navigable_positions.npy"
                )
                if npy_fp.exists():
                    self._navigable_positions[bci] = np.load(npy_fp)

        self.scene.set_ambient_light([0.3, 0.3, 0.3])

        # merge actors into one
        self.bg = Actor.create_from_entities(
            bgs,
            scene=self.scene,
            scene_idxs=torch.arange(self.env.num_envs, dtype=int),
            shared_name="scene_background",
        )

    def initialize(self, env_idx):

        if self.env.robot_uids == "fetch":
            agent: Fetch = self.env.agent
            rest_keyframe = agent.keyframes["rest"]
            agent.reset(rest_keyframe.qpos)

            agent.robot.set_pose(
                Pose.create_from_pq(
                    p=[
                        [*FETCH_BUILD_CONFIG_IDX_TO_START_POS[bci], 0.02]
                        for bci in self.build_config_idxs
                    ]
                )
            )

            # For the purposes of physical simulation, we disable collisions between the Fetch robot and the scene background
            self.disable_fetch_move_collisions(self.bg)
        else:
            raise NotImplementedError(self.env.robot_uids)

        for obj, pose in self._default_object_poses:
            obj.set_pose(pose)
            if isinstance(obj, Articulation):
                # note that during initialization you may only ever change poses/qpos of objects in scenes being reset
                obj.set_qpos(obj.qpos[0] * 0)

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
    def navigable_positions(self) -> List[np.ndarray]:
        return [self._navigable_positions[bci] for bci in self.build_config_idxs]
