from pathlib import Path
from collections import OrderedDict
from typing import Dict, Union

import numpy as np
import torch
import torch.random
from transforms3d.euler import euler2quat
import sapien
import sapien.physx as physx
import sapien.render

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.scene_builder.table.table_scene_builder import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

@register_env("BookToShelf-v0", max_episode_steps=50)
class BookToShelfEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = [
        "sparse",
        "none",
    ]  # TODO add a denser reward for this later
    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _sensor_configs(self):
        pose = look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _human_render_camera_configs(self):
        pose = look_at([1, -0.4, 0.6], [0.4, -1, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.book = actors.build_actor_ai2_helper(
            model_id="Book_14",
            scene=self._scene,
            name="book",
            kinematic=False,
            set_object_on_ground=False,
        )

        self.shelf = actors.build_actor_ai2_helper(
            model_id="Shelving_Unit_307_1",
            scene=self._scene,
            name="shelf",
            kinematic=True,
            set_object_on_ground=False,
        )

    def _initialize_episode(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz_r = xyz
            xyz_s = xyz

            # robot pose
            xyz_r[..., 0] = -0.5
            xyz_r[..., 1] = -1
            xyz_r[..., 2] = 0
            robot_pose = Pose.create_from_pq(p=xyz_r, q = euler2quat(0, 0, 0))
            self.agent.robot.set_pose(robot_pose)

            # book pose
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 1] -= 0.95 
            xyz[..., 2] += 0.075
            q = euler2quat(0, np.pi/2, np.pi/2)
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.book.set_pose(obj_pose)

            # shelf pose
            xyz_s[..., 0] = -0.2
            xyz_s[..., 1] = -1.35
            xyz_s[..., 2] = -0.425
            q = euler2quat(0, 0, np.pi)
            obj_pose = Pose.create_from_pq(p=xyz_s, q=q)
            self.shelf.set_pose(obj_pose)
            self.shelf_aabb = (
                self.shelf._objs[0]
                .find_component_by_type(sapien.render.RenderBodyComponent)
                .compute_global_aabb_tight()
            )

    def check_bbox_in_shelf(self):
        book_aabb = (
            self.book._objs[0]
            .find_component_by_type(sapien.render.RenderBodyComponent)
            .compute_global_aabb_tight()
        )
        max_x = self.shelf_aabb[1, 0]
        max_y = self.shelf_aabb[1, 1]
        max_z = self.shelf_aabb[1, 2]

        min_x = self.shelf_aabb[0, 0]
        min_y = self.shelf_aabb[0, 1]
        min_z = self.shelf_aabb[0, 2]

        book_aabb = torch.from_numpy(book_aabb)
        x_flag_max = max_x + 0.05 >= book_aabb[1, 0]
        y_flag_max = max_y + 0.05 >= book_aabb[1, 1]
        z_flag_max = max_z >= book_aabb[1, 2]

        x_flag_min = min_x - 0.05 <= book_aabb[0, 0]
        y_flag_min = min_y - 0.05 <= book_aabb[0, 1]
        z_flag_min = min_z <= book_aabb[0, 2]

        return (
            x_flag_max & y_flag_max & z_flag_max &
            x_flag_min & y_flag_min & z_flag_min
        )

    def evaluate(self):
        is_book_placed = self.check_bbox_in_shelf()

        return {
            "success": is_book_placed,
        }

    def _get_obs_extra(self, info: Dict):
        obs = OrderedDict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=self.book.pose.raw_pose,
            )
        return obs