from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill import logger
import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.building.actors.common import build_container_grid


@register_env("PickAndPlace-v1", max_episode_steps=500)
class PickAndPlaceEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda_wristcam"]
    SUPPORTED_REWARD_MODES = ["none", "sparse"]
    agent: Union[PandaWristCam]
    cube_half_size = 0.02
    goal_thresh = 0.05

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.container_grid, self.goal_sites = build_container_grid(
            self.scene,
            initial_pose=sapien.Pose(p=[0.0, 0.2, 0.04], q=[1, 0, 0, 0]),
            size=0.25,
            height=0.05,
            thickness=0.01,
            color=(0.8, 0.6, 0.4),
            name="container_grid",
            n=2,
            m=2,
        )
        self.red_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="red_cube",
            initial_pose=sapien.Pose(p=[0,0,0.1])
        )

        self.green_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0, 1, 0, 1],
            name="green_cube",
            initial_pose=sapien.Pose(p=[0,0,0.5])
        )

        self.blue_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0, 0, 1, 1],
            name="blue_cube",
            initial_pose=sapien.Pose(p=[0,0,1.0])
        )

        self.yellow_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 1, 0, 1],
            name="yellow_cube",
            initial_pose=sapien.Pose(p=[0,0,1.5])
        )

        self.cubes = [self.red_cube, self.green_cube, self.blue_cube, self.yellow_cube]
        self._hidden_objects.extend(self.goal_sites)
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            env_count = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Initialize container grid
            container_pose = sapien.Pose(p=[0.0, 0.2, 0.04], q=[1, 0, 0, 0])
            self.container_grid.set_pose(container_pose)

            region = [[0.05, -0.15], [0.09, -0.1]]
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            min_distance = 0.06

            # Randomize cube positions
            for i, cube in enumerate(self.cubes):
                while True:
                    xyz = torch.zeros((env_count, 3))
                    sampler = randomization.UniformPlacementSampler(
                        bounds=region, batch_size=env_count, device=self.device
                    )                
                    cube_xy = torch.rand((env_count, 2)) * 0.4 - 0.4 + sampler.sample(radius, 100)
                    xyz[:, :2] = cube_xy
                    xyz[:, 2] = 0.04
                    qs = randomization.random_quaternions(
                        env_count,
                        lock_x=True,
                        lock_y=True,
                        lock_z=False,
                    )
                    cube.set_pose(Pose.create_from_pq(p=xyz, q=qs))

                    overlap = False
                    for j in range(i):
                        other_cube = self.cubes[j]
                        other_xyz = other_cube.pose.p
                        distance = torch.linalg.norm(xyz - other_xyz, axis=1)
                        if torch.any(distance < min_distance):
                            overlap = True
                            break

                    if not overlap:
                        break

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = {"tcp_pose": self.agent.tcp.pose.raw_pose}
        for goal_site in self.goal_sites:
            obs[f"{goal_site.name}_pose"] = goal_site.pose.p

        for i, obj in enumerate(self.cubes):
            obs[f"{obj.name}_pose"] = obj.pose.raw_pose
            if "state" in self.obs_mode:
                pass
                obs[f"{obj.name}_to_goal_pos"] = self.goal_sites[i].pose.p - obj.pose.p
                obs[f"tcp_to_{obj.name}_pos"] = obj.pose.p - self.agent.tcp.pose.p

        return {}


    def evaluate(self):
        results = dict()
        total_distance_to_goal = torch.tensor(0.0, device=self.device)
        all_placed = torch.tensor(True, device=self.device)
        any_grasped = torch.tensor(False, device=self.device)
        # only count success after all objects grasped and placed

        for i, goal_site in enumerate(self.goal_sites):
            obj = self.cubes[i]
            obj_name = obj.name

            distance_to_goal = torch.linalg.norm(goal_site.pose.p - obj.pose.p, axis=1)
            is_placed =  (
                distance_to_goal
                <= self.goal_thresh
            )
            is_grasped = self.agent.is_grasping(obj)

            results[f"{obj_name}_distance_to_goal"] = distance_to_goal
            results[f"is_{obj_name}_placed"] = is_placed
            results[f"is_{obj_name}_grasped"] = is_grasped

            all_placed = torch.logical_and(all_placed, is_placed)
            any_grasped = torch.logical_or(any_grasped, is_grasped)


        # Success is defined as all cubes being placed and none being grasped
        results["success"] = torch.logical_and(all_placed, torch.logical_not(any_grasped))

        # Reward for the robot being static
        results["is_robot_static"] = self.agent.is_static(0.2)

        return results
