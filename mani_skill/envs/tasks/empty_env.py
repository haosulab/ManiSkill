from typing import Dict

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("Empty-v1", max_episode_steps=200000)
class EmptyEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none"]
    """
    This is just a dummy environment for showcasing robots in a empty scene
    """

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([1.25, -1.25, 1.5], [0.0, 0.0, 0.2])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at([1.25, -1.25, 1.5], [0.0, 0.0, 0.2])
        pose = sapien_utils.look_at([6, 12.5, 1.5], [6, 12, 1.0])
        return CameraConfig("render_camera", pose, 2048, 2048, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose())

    def _load_scene(self, options: dict):
        # self.ground = build_ground(self.scene)
        # self.ground.set_collision_group_bit(group=2, bit_idx=30, bit=1)
        self.bg = self.scene.create_actor_builder()
        import glob
        import json

        # current export tool does not set right texture paths for some room materials for some reason
        # fix is run this for each room and modify the texture paths to be textures/room_name/*.png...
        # cat /home/stao/work/external/infinigen/outputs/sapien/export_scene.blend/export_scene.obj | grep -v '^[vf]' | grep -v '^vt' | grep -v '^s' | grep kitchen -A 2
        # python -m infinigen.tools.export --input_folder outputs/indoors/coarse/ --output_folder outputs/sapien -f obj -r 1024
        # need new export tool. The tool needs to export the poses of all objects and export each as a separate part (best as separate files) so we can load each
        # object individually and simulate it. Also load lighting info details.
        import os

        from transforms3d.euler import euler2quat

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
                actor = self.scene.create_actor_builder()
                actor.add_visual_from_file(
                    filename=obj_file,
                )
                rot_x = scene_metadata[subfolder]["rotation"][0]
                rot_y = scene_metadata[subfolder]["rotation"][1]
                rot_z = scene_metadata[subfolder]["rotation"][2]
                actor.initial_pose = sapien.Pose(
                    p=scene_metadata[subfolder]["position"],
                    q=euler2quat(rot_x, rot_y, rot_z, axes="rxyz"),
                ) * sapien.Pose(q=euler2quat(np.pi / 2, 0, 0, axes="rxyz"))
                actor.build_static(name=subfolder)
        # self.bg.add_visual_from_file(filename="/home/stao/work/external/infinigen/outputs/sapien_i/export_scene.blend/dining-room_0_0_wall/dining-room_0_0_wall.obj", pose=sapien.Pose(p=[0, 0, 0], q=euler2quat(np.pi/2, 0*np.pi/2, 0)))
        # self.bg.add_visual_from_file(filename="/home/stao/work/external/infinigen/outputs/sapien/export_scene.blend/export_scene.obj", pose=sapien.Pose(p=[0, 0, 0], q=euler2quat(np.pi/2, 0*np.pi/2, 0)))
        # self.bg.build_static(name="bg")

    def _load_lighting(self, options: Dict):
        super()._load_lighting(options)
        self.scene.add_point_light(
            position=[6.016, 12.0, 2.7], color=[1, 1, 1], shadow=True
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        pass

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: Dict):
        return dict()
