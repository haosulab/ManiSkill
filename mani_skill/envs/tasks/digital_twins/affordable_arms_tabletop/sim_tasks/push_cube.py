from typing import Any, Dict

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots.koch.koch import Koch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor
from mani_skill.utils.structs.pose import Pose

from .sim4real_base_env import Sim4RealBaseEnv


# grab cube and return to rest keyframe
@register_env("Affordable-PushCube-v1", max_episode_steps=50)
class PushCubeEnv(Sim4RealBaseEnv):
    table_edge_x = 0
    # While this is the goal x value, success is
    # pushing the cube to any x value gt the starting x position + 3 cm
    goal_pos_x = table_edge_x + 0.35

    # Task DR
    spawn_box_half_size = 0.1 / 2

    cube_size_mean = 0.017 / 2
    cube_size_std = 7e-4 / 2

    cube_friction_mean = 0.3
    cube_friction_std = 0.1

    rand_cube_color = True

    def __init__(
        self,
        *args,
        robot_uids="koch-v1.1",
        robot_init_qpos_noise=0.02 * 2,
        num_envs=1,
        cam_rand_on_step=True,
        keyframe_id=None,
        enable_shadow=False,
        **kwargs,
    ):
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            num_envs=num_envs,
            cam_rand_on_step=cam_rand_on_step,
            keyframe_id=keyframe_id,
            enable_shadow=enable_shadow,
            **kwargs,
        )

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        cubes = []
        half_sizes = []
        # CUBE DR: size, friction, and color
        # ROBOT DR: color
        sampled_sizes = torch.normal(
            mean=torch.ones(self.num_envs) * self.cube_size_mean,
            std=torch.ones(self.num_envs) * self.cube_size_std,
        )
        sampled_frictions = torch.normal(
            mean=torch.ones(self.num_envs) * self.cube_friction_mean,
            std=torch.ones(self.num_envs) * self.cube_friction_std,
        )
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            half_size = sampled_sizes[i].item()
            friction = sampled_frictions[i].clip(0.1, 0.5).item()
            half_sizes.append(half_size)
            material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0,
            )
            builder.add_box_collision(
                half_size=[half_size] * 3, material=material, density=500  # 25
            )
            color = list(torch.rand(3))
            builder.add_box_visual(
                half_size=[half_size] * 3,
                material=sapien.render.RenderMaterial(
                    base_color=[color[0], color[1], color[2], 1],
                ),
            )
            # setting new pose for cube
            builder.initial_pose = sapien.Pose(p=[0, 1, 2 * half_size])
            builder.set_scene_idxs([i])
            cube = builder.build(f"cube_{i}")
            cubes.append(cube)
        self.cube = Actor.merge(cubes, "cube")
        self.cube_half_sizes = torch.tensor(half_sizes).to(self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            xyz = torch.zeros((b, 3))
            spawn_box_pos = self.agent.robot.pose.p + torch.tensor([0.225, 0, 0])

            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.spawn_box_half_size * 2
                - self.spawn_box_half_size
            )
            xyz[:, :2] += spawn_box_pos[env_idx, :2]
            xyz[:, 2] = self.cube_half_sizes[env_idx]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))
            self.cube_start_pos = self.cube.pose.p

            keyframe = torch.tensor(self.agent.keyframes["to_push"].qpos)
            qpos = torch.ones(b, keyframe.shape[-1]) * keyframe.view(1, -1)
            qpos_noise = torch.normal(
                0, torch.ones(b, keyframe.shape[-1]) * self.robot_init_qpos_noise
            )  # * 0
            self.agent.robot.set_qpos(qpos + qpos_noise)

    def _get_obs_extra(self, info: Dict):
        obs = dict()
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p
                - (self.agent.tcp.pose.p + self.agent.tcp2.pose.p) / 2,
                tcp_pose=self.agent.tcp.pose.raw_pose,
                tcp2_pose=self.agent.tcp2.pose.raw_pose,
                cube_side_length=self.cube_half_sizes * 2,
                goal_pos=self.goal_pos_x
                * torch.ones(self.num_envs, 1, device=self.device),
            )

        # automatically included state info
        qpos = self.agent.robot.get_qpos().clone()
        obs.update(
            qpos=qpos,
            target_qpos=self.agent.controller._target_qpos.clone(),
        )
        return obs

    def evaluate(self):
        touching_table = self.agent._compute_undesired_contacts(
            self.table_scene.table
        ).float()
        touching_cube = self.agent._compute_undesired_contacts(self.cube).float()
        # reward same if cube pushed past goal x position, toward middle of table
        # reward for pushing past goal (2.5 cm)
        cube_to_goal_dist = ((self.goal_pos_x + 0.025) - self.cube.pose.p[..., 0]).clip(
            0, np.inf
        )
        # success = cube_to_goal_dist <= 0.025
        success = self.cube.pose.p[..., 0] > (self.cube_start_pos[..., 0] + 0.03)
        return {
            "touching_table": touching_table,
            "touching_cube": touching_cube,
            "cube_to_goal_dist": cube_to_goal_dist,
            "success": success,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = (self.agent.tcp.pose.p + self.agent.tcp2.pose.p) / 2

        reward = 0

        # TODO (xhin): reward to not move joint -2, believe there is a distribution shift, fix and remove
        reward += 0.2 * (
            1 - torch.tanh((self.agent.robot.qpos[..., -2] + np.pi / 2).abs())
        )

        # reach tcp to object reward
        goal_touch_pos = self.cube.pose.p
        goal_touch_pos[..., -1] = 0
        tcp_to_obj_dist = torch.linalg.norm(
            (goal_touch_pos - tcp_pos),
            axis=-1,
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward += 2 * reaching_reward

        # additional reward to keep robot flat against table
        reward += 1 - torch.tanh(10 * tcp_pos[..., -1].abs())

        # reach cube to goal reward
        reward += 2 * reaching_reward * (1 - torch.tanh(5 * info["cube_to_goal_dist"]))

        # negative reward for gripper passing cube without pushing
        gripper_past_cube = self.agent.tcp.pose.p[..., 0] > self.cube.pose.p[..., 0]
        passed_rew = gripper_past_cube * torch.tanh(
            10 * (self.agent.tcp.pose.p[..., 0] - self.cube.pose.p[..., 0]).abs()
        )
        reward -= 2 * passed_rew

        return reward - 0.2 * info["touching_table"]

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.4
