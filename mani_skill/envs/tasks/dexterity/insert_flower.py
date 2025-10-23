import os
from typing import Any, Union

import numpy as np
import sapien.physx as physx
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.robots import FloatingAbilityHandRight
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


@register_env(
    "InsertFlower-v1", max_episode_steps=300, asset_download_ids=["oakink-v2"]
)
class InsertFlowerEnv(BaseEnv):
    agent: Union[FloatingAbilityHandRight]
    _clearance = 0.003
    hand_init_height = 0.25
    flower_spawn_half_size = 0.05
    asset_path = f"{ASSET_DIR}/tasks/oakink-v2/align_ds"

    def __init__(
        self,
        *args,
        robot_uids="floating_ability_hand_right",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.target_area = {"min": [-0.3, -0.25, 0.25], "max": [-0.2, -0.15, 0.35]}

        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

        with torch.device(self.device):
            self.prev_unit_vector = torch.zeros((self.num_envs, 3))
            self.cum_rotation_angle = torch.zeros((self.num_envs,))

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=self.num_envs * max(1024, self.num_envs) * 8,
                max_rigid_patch_count=self.num_envs * max(1024, self.num_envs) * 2,
                found_lost_pairs_capacity=2**26,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.15, 0, 0.45], target=[-0.1, 0, self.hand_init_height]
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.2, 0.4, 0.6], [0.0, 0.0, 0.3])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # === Load Vase (Static) ===
        vase_builder = self.scene.create_actor_builder()
        vase_visual_mesh_file = os.path.join(
            self.asset_path, "O02@0080@00001/model.obj"
        )
        vase_collision_mesh_file = os.path.join(
            self.asset_path, "O02@0080@00001/model.obj.coacd.ply"
        )
        vase_builder.add_visual_from_file(vase_visual_mesh_file)
        vase_builder.add_multiple_convex_collisions_from_file(vase_collision_mesh_file)
        vase_builder.initial_pose = Pose.create_from_pq(
            [-0.2509, -0.2027, 0.102 + 0.001], [0.8712, 0.0069, 0.0082, 0.4908]
        )

        self.vase = vase_builder.build_static(name="vase")

        # === Load Flower (Dynamic) ===
        flower_builder = self.scene.create_actor_builder()
        flower_mesh_file = os.path.join(self.asset_path, "O02@0081@00001/model.obj")
        flower_collision_file = os.path.join(
            self.asset_path, "O02@0081@00001/model.obj.coacd.ply"
        )
        flower_builder.add_visual_from_file(flower_mesh_file)

        flower_material = physx.PhysxMaterial(
            static_friction=1, dynamic_friction=1, restitution=1
        )
        flower_builder.add_multiple_convex_collisions_from_file(
            flower_collision_file, density=200, material=flower_material
        )

        flower_builder.initial_pose = Pose.create_from_pq(
            [-0.242, 0.0, 0.015 + 0.001], [-0.352413, -0.258145, -0.635074, 0.637062]
        )
        self.initial_flower_pos = torch.tensor(
            [-0.242, 0.0, 0.015 + 0.001], device=self.device
        )
        self.initial_flower_quat = torch.tensor(
            [-0.352413, -0.258145, -0.635074, 0.637062], device=self.device
        )
        self.flower = flower_builder.build(name="flower")
        self.target_area_box = list(self.target_area.values())
        # Convert target_area into tensor for fast computation
        self.target_area_box = torch.tensor(
            self.target_area_box, device=self.device, dtype=torch.float32
        ).view(2, 3)

    def _after_reconfigure(self, options: dict):
        pass

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_actors(env_idx)
        self._initialize_agent(env_idx)

    def _initialize_actors(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)

            self.table_scene.initialize(env_idx)

            flower_pos = (
                torch.rand((b, 3)) * self.flower_spawn_half_size * 2
                - self.flower_spawn_half_size
                + self.initial_flower_pos
            )
            flower_pose = Pose.create_from_pq(flower_pos, self.initial_flower_quat)
            self.flower.set_pose(flower_pose)

    def _initialize_agent(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            dof = self.agent.robot.dof
            if isinstance(dof, torch.Tensor):
                dof = dof[0]
            init_qpos = torch.zeros((b, dof))
            self.agent.reset(init_qpos)
            self.agent.robot.set_pose(
                Pose.create_from_pq(
                    torch.tensor([0.0, 0, self.hand_init_height]),
                    torch.tensor(
                        [
                            0,
                            0.707,
                            0,
                            -0.707,
                        ]
                    ),
                )
            )

    def _get_obs_extra(self, info: dict):
        return {}

    def evaluate(self, **kwargs) -> dict:
        object_pos = self.flower.pose.p

        # Check if the object is within the specified bounds
        is_within = torch.logical_and(
            torch.all(object_pos >= self.target_area_box[0], dim=-1),  # min bounds
            torch.all(object_pos <= self.target_area_box[1], dim=-1),  # max bounds
        )

        return {"success": is_within}

    def compute_dense_reward(self, obs: Any, action: Array, info: dict) -> float:
        object_pos = self.flower.pose.p
        dist_outside = torch.max(
            torch.max(
                self.target_area_box[0] - object_pos, torch.zeros_like(object_pos)
            ),  # lower bound
            torch.max(
                object_pos - self.target_area_box[1], torch.zeros_like(object_pos)
            ),  # upper bound
        )
        reward = torch.exp(-5 * torch.norm(dist_outside)).reshape(-1)

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4.0


if __name__ == "__main__":
    print(InsertFlowerEnv.asset_path)
