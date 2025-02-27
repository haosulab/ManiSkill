import glob
import json
import os
from pathlib import Path
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
    builds_lighting = True

    def build(self, build_config_idxs: Union[int, List[int]] = None):
        # if build_config_idxs is None:
        # build_config_idxs = [0]
        base_path = "/home/stao/work/external/infinigen/outputs/sapien_i_metadata/export_scene.blend"
        scene_metadata = json.load(open(os.path.join(base_path, "metadata.json")))
        # load each object in the scene
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
                actor = self.scene.create_actor_builder()
                scene_metadata["objects"][subfolder]["rotation"][0]
                scene_metadata["objects"][subfolder]["rotation"][1]
                scene_metadata["objects"][subfolder]["rotation"][2]
                # infinigen exports assets oriented such that the natural "front" of the object is not at the identity pose
                # we fix it here so that "front" is front when object initial pose is identity.
                actor.add_visual_from_file(
                    filename=obj_file,
                    # pose=sapien.Pose(q=euler2quat(rot_x, rot_y, -rot_z, axes="sxyz"))
                    pose=sapien.Pose(q=euler2quat(np.pi / 2, 0, 0, axes="sxyz")),
                )
                actor.initial_pose = sapien.Pose(
                    p=scene_metadata["objects"][subfolder]["position"],
                    # q=euler2quat(rot_x, rot_y, rot_z, axes="sxyz"),
                )
                actor.build_static(name=subfolder)
        # TODO can we export the env maps?
        for sub_scene in self.scene.sub_scenes:
            sub_scene.set_environment_map(
                str(
                    (
                        Path(__file__).parent
                        / "../replicacad/autumn_field_puresky_4k.hdr"
                    ).absolute()
                )
            )
        for light_metadata in scene_metadata["lighting"].values():
            # import ipdb; ipdb.set_trace()
            light_type = light_metadata["type"]
            if light_type == "point":
                # light = self.scene.create_actor_builder()
                # self.scene.add_area_light_for_ray_tracing(
                #     pose=sapien.Pose(p=light_metadata["position"], q=sapien.Pose(q=euler2quat(np.pi / 2, 0, 0, axes="sxyz"))),
                #     color=np.array(light_metadata["color"]) * float(light_metadata["energy"]) / 50,
                #     half_width=5, #light_metadata["shadow_soft_size"],
                #     half_height=5, #light_metadata["shadow_soft_size"],
                # )
                self.scene.add_point_light(
                    position=np.array(light_metadata["position"])
                    + np.array([0, 0, -0.05]),
                    color=np.array(light_metadata["color"])
                    * float(light_metadata["energy"])
                    / 100,
                    # energy=light_metadata["energy"],
                    # shadow_soft_size=light_metadata["shadow_soft_size"],
                )

    def initialize(self, env_idx: torch.Tensor):
        pass
