import glob
import json
import os
from typing import List, Union

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.scene_builder.registration import register_scene_builder


@register_scene_builder("InfinigenIndoors")
class InfinigenIndoorsSceneBuilder(SceneBuilder):
    def build(self, build_config_idxs: Union[int, List[int]] = None):
        # if build_config_idxs is None:
        # build_config_idxs = [0]
        base_path = "/home/stao/work/external/infinigen/outputs/sapien_i_metadata/export_scene.blend"
        scene_metadata = json.load(open(os.path.join(base_path, "metadata.json")))
        for subfolder in os.listdir(base_path):
            subfolder_path = os.path.join(base_path, subfolder)
            if os.path.isdir(subfolder_path):
                # Find the .obj file in this subfolder
                obj_files = glob.glob(os.path.join(subfolder_path, "*.obj"))
                # Check for corresponding .mtl file
                mtl_files = glob.glob(os.path.join(subfolder_path, "*.mtl"))
                for mtl_file in mtl_files:
                    # Read MTL file
                    with open(mtl_file, "r") as f:
                        lines = f.readlines()

                    # Filter out aniso/anisor lines which don't do anything it seems
                    filtered_lines = [
                        line
                        for line in lines
                        if not line.strip().startswith(("aniso", "anisor"))
                    ]

                    # Write back filtered content
                    with open(mtl_file, "w") as f:
                        f.writelines(filtered_lines)
                if len(obj_files) != 1:
                    continue
                obj_file = obj_files[0]
                # if "chair" not in subfolder.lower():
                #     continue
                actor = self.scene.create_actor_builder()
                rot_x = scene_metadata[subfolder]["rotation"][0]
                rot_y = scene_metadata[subfolder]["rotation"][1]
                rot_z = scene_metadata[subfolder]["rotation"][2]
                # infinigen exports assets oriented such that the natural "front" of the object is not at the identity pose
                # we fix it here so that "front" is front when object initial pose is identity.
                actor.add_visual_from_file(
                    filename=obj_file,
                    pose=sapien.Pose(q=euler2quat(rot_x, rot_y, rot_z, axes="sxyz"))
                    * sapien.Pose(q=euler2quat(np.pi / 2, 0, 0, axes="sxyz")),
                )
                actor.initial_pose = sapien.Pose(
                    p=scene_metadata[subfolder]["position"],
                    q=euler2quat(rot_x, rot_y, rot_z, axes="sxyz"),
                )
                actor.build_static(name=subfolder)

    def initialize(self, env_idx: torch.Tensor):
        pass
