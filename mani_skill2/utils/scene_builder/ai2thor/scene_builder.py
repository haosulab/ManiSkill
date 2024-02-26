"""
SceneBuilder for the AI2Thor scenes, using configurations and assets stored in https://huggingface.co/datasets/hssd/ai2thor-hab

"""


import json
import os.path as osp
from pathlib import Path
from typing import List

import numpy as np
import sapien
import sapien.core as sapien
import sapien.physx as physx
import transforms3d
from tqdm import tqdm

from mani_skill2 import ASSET_DIR
from mani_skill2.utils.scene_builder import SceneBuilder

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


class AI2THORBaseSceneBuilder(SceneBuilder):
    """
    The Base AI2THOR scene builder class. Subclasses
    """

    scene_dataset: str

    def __init__(self):
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

        super().__init__()

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

    def build(
        self, scene: sapien.Scene, scene_id=0, convex_decomposition="none", **kwargs
    ):
        # save scene and movable objects when building scene
        self._scene_objects: List[sapien.Entity] = []
        self._movable_objects: List[sapien.Entity] = []

        scene_cfg = self._scene_configs[scene_id]

        dataset = SCENE_SOURCE_TO_DATASET[scene_cfg.source]
        with open(osp.join(dataset.dataset_path, scene_cfg.config_file), "rb") as f:
            scene_json = json.load(f)

        bg_path = str(
            Path(ASSET_DIR)
            / "scene_datasets/ai2thor/ai2thor-hab/assets"
            / f"{scene_json['stage_instance']['template_name']}.glb"
        )
        builder = scene.create_actor_builder()
        builder.add_visual_from_file(bg_path)
        builder.add_nonconvex_collision_from_file(bg_path)
        bg = builder.build_kinematic(name="scene_background")
        q = transforms3d.quaternions.axangle2quat(
            np.array([1, 0, 0]), theta=np.deg2rad(90)
        )
        if self.scene_dataset == "ProcTHOR":
            # for some reason the scene needs to rotate around y-axis by 90 degrees for ProcTHOR scenes from hssd dataset
            q = transforms3d.quaternions.qmult(
                q,
                transforms3d.quaternions.axangle2quat(
                    np.array([0, -1, 0]), theta=np.deg2rad(90)
                ),
            )
        bg.set_pose(sapien.Pose(q=q))

        global_id = 0
        for object in tqdm(scene_json["object_instances"][:]):
            model_path = (
                Path(ASSET_DIR)
                / "scene_datasets/ai2thor/ai2thorhab-uncompressed/assets"
                / f"{object['template_name']}.glb"
            )
            actor_id = f"{object['template_name']}_{global_id}"
            global_id += 1
            builder = scene.create_actor_builder()
            builder.add_visual_from_file(str(model_path))
            if self._should_be_kinematic(object["template_name"]):
                position = [
                    object["translation"][0],
                    -object["translation"][2],
                    object["translation"][1] + 0,
                ]
                builder.add_nonconvex_collision_from_file(str(model_path))
                actor = builder.build_kinematic(name=actor_id)
            else:
                position = [
                    object["translation"][0],
                    -object["translation"][2],
                    object["translation"][1] + 0.005,
                ]
                builder.add_multiple_convex_collisions_from_file(
                    str(model_path), decomposition=convex_decomposition
                )
                actor = builder.build(name=actor_id)
            self._scene_objects.append(actor)

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
            pose = sapien.Pose(p=position, q=q)
            actor.set_pose(pose)

        # get movable objects
        self._movable_objects = [
            obj
            for obj in self.scene_objects
            if not obj.find_component_by_type(
                physx.PhysxRigidDynamicComponent
            ).kinematic
        ]

    @property
    def scene_configs(self):
        return self._scene_configs

    @property
    def scene_objects(self):
        return self._scene_objects

    @property
    def movable_objects(self):
        return self._movable_objects
