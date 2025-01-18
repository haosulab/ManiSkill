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
@register_env("Affordable-OddCubeOut-v1", max_episode_steps=50)
class OddCubeOutEnv(Sim4RealBaseEnv):
    table_edge_x = -0.737

    # Task DR
    # spawn_box_half_size_y = 0.05 / 2
    # spawn_box_half_size_x = 0.1 / 2
    spawn_box_half_size_y = 0.03 / 2
    spawn_box_half_size_x = 0.04 / 2
    spawn_box_x_offset = 0.25
    # left_spawn_box_pos = torch.tensor([table_edge_x + spawn_box_x_offset, -0.07, 0])
    # middle_spawn_box_pos = torch.tensor([table_edge_x + spawn_box_x_offset, 0, 0])
    # right_spawn_box_pos = torch.tensor([table_edge_x + spawn_box_x_offset, 0.07, 0])
    left_spawn_box_pos = torch.tensor([table_edge_x + spawn_box_x_offset, -0.05, 0])
    middle_spawn_box_pos = torch.tensor([table_edge_x + spawn_box_x_offset, 0, 0])
    right_spawn_box_pos = torch.tensor([table_edge_x + spawn_box_x_offset, 0.05, 0])

    cube_size_mean = 0.017 / 2
    cube_size_std = 7e-4 / 2

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

    def make_cubes(self, colors, half_sizes, init_z, obj_name):
        cubes = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            half_size = half_sizes[i].item()
            builder.add_box_collision(half_size=[half_size] * 3, density=500)
            color = colors[i]
            builder.add_box_visual(
                half_size=[half_size] * 3,
                material=sapien.render.RenderMaterial(
                    base_color=[color[0].item(), color[1].item(), color[2].item(), 1],
                ),
            )
            # setting new pose for cube
            builder.initial_pose = sapien.Pose(p=[0, 0, init_z])
            builder.set_scene_idxs([i])
            cube = builder.build(obj_name + "_" + str(i))
            cubes.append(cube)
        cube = Actor.merge(cubes, "cube")
        return cube

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        # CUBE DR: size, and color
        # sizes of cubes within each scene should be the same
        sampled_sizes = torch.normal(
            mean=torch.ones(self.num_envs) * self.cube_size_mean,
            std=torch.ones(self.num_envs) * self.cube_size_std,
        )

        distraction_cube_color = torch.rand(self.num_envs, 3)
        goal_cube_color = torch.rand(self.num_envs, 3)

        color_dist = torch.linalg.norm(distraction_cube_color - goal_cube_color, dim=-1)
        similar_colors = (
            color_dist <= 0.5
        )  # if color is (0.5,0.5,0.5), max dist achievable is sqrt(3*0.5^2) = 0.87
        while similar_colors.any():
            goal_cube_color[similar_colors] = torch.rand(similar_colors.sum(), 3)
            color_dist = torch.linalg.norm(
                distraction_cube_color - goal_cube_color, dim=-1
            )
            similar_colors = color_dist <= 0.5

        self.first_d_cube = self.make_cubes(
            distraction_cube_color, sampled_sizes, 0.1, "distraction_1"
        )
        self.second_d_cube = self.make_cubes(
            distraction_cube_color, sampled_sizes, 0.2, "distraction_2"
        )
        self.goal_cube = self.make_cubes(
            goal_cube_color, sampled_sizes, 0.3, "goal_cube"
        )

        self.cube_half_sizes = sampled_sizes.to(self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)

            xyz = torch.zeros((b, 3, 3))
            xyz[..., 0] = (
                torch.rand((b, 3)) * self.spawn_box_half_size_x * 2
                - self.spawn_box_half_size_x
            )
            xyz[..., 1] = (
                torch.rand((b, 3)) * self.spawn_box_half_size_y * 2
                - self.spawn_box_half_size_y
            )
            xyz[:, 0, :] += self.left_spawn_box_pos.to(self.device)
            xyz[:, 1, :] += self.middle_spawn_box_pos.to(self.device)
            xyz[:, 2, :] += self.right_spawn_box_pos.to(self.device)
            xyz[:, :, -1] = self.cube_half_sizes[env_idx].view(b, 1)
            qs = randomization.random_quaternions(3 * b, lock_x=True, lock_y=True).view(
                b, 3, 4
            )

            indices = torch.vstack([torch.randperm(3) for _ in range(b)]).T

            self.first_d_cube.set_pose(
                Pose.create_from_pq(
                    p=xyz[torch.arange(b), indices[0]].view(b, 3), q=qs[:, 0].view(b, 4)
                )
            )
            self.second_d_cube.set_pose(
                Pose.create_from_pq(
                    p=xyz[torch.arange(b), indices[1]].view(b, 3), q=qs[:, 1].view(b, 4)
                )
            )
            self.goal_cube.set_pose(
                Pose.create_from_pq(
                    p=xyz[torch.arange(b), indices[2]].view(b, 3), q=qs[:, 2].view(b, 4)
                )
            )

            keyframe = torch.tensor(self.agent.keyframes["closed_gripper"].qpos)
            qpos = torch.ones(b, keyframe.shape[-1]) * keyframe.view(1, -1)
            qpos_noise = torch.normal(
                0, torch.ones(b, keyframe.shape[-1]) * self.robot_init_qpos_noise
            )  # * 0
            self.agent.robot.set_qpos(qpos + qpos_noise)

    def _get_obs_extra(self, info: Dict):
        obs = dict()
        if "state" in self.obs_mode:
            return NotImplementedError(
                "Warning: attempting to use state information for vision reasoning task"
            )

        # automatically included state info
        qpos = self.agent.robot.get_qpos().clone()
        obs.update(
            qpos=qpos,
            target_qpos=self.agent.controller._target_qpos.clone(),
        )
        return obs

    def evaluate(self):
        touch_table = self.agent._compute_undesired_contacts(
            self.table_scene.table
        ).float()
        touch_first_d_cube = self.agent._compute_undesired_contacts(
            self.first_d_cube, threshold=1e-3
        )
        touch_second_d_cube = self.agent._compute_undesired_contacts(
            self.second_d_cube, threshold=1e-3
        )
        touch_distraction = torch.logical_or(
            touch_first_d_cube, touch_second_d_cube
        ).float()
        touch_goal = self.agent._compute_undesired_contacts(
            self.goal_cube, threshold=1e-3
        )
        return {
            "touch_table": touch_table,
            "touch_distraction": touch_distraction,
            "touch_goal": touch_goal,
            "success": touch_goal,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = (self.agent.tcp.pose.p + self.agent.tcp2.pose.p) / 2

        # reward to not move the gripper - not necessary for task, greatly reduces exploration
        reward = 0.2 * (1 - torch.tanh((self.agent.robot.qpos[..., -1]).abs()))

        # TODO (xhin): reward to not move joint -2, believe there is a distribution shift, fix and remove
        reward += 0.2 * (
            1 - torch.tanh((self.agent.robot.qpos[..., -2] + np.pi / 2).abs())
        )

        # negative reward for touching wrong cube 1
        distract_1_touch_pos = self.first_d_cube.pose.p
        tcp_to_obj_dist = torch.linalg.norm(
            (distract_1_touch_pos - tcp_pos),
            axis=-1,
        )
        reward -= 0.2 * (1 - torch.tanh(5 * tcp_to_obj_dist))

        # negative reward for touching wrong cube 2
        distract_1_touch_pos = self.first_d_cube.pose.p
        tcp_to_obj_dist = torch.linalg.norm(
            (distract_1_touch_pos - tcp_pos),
            axis=-1,
        )
        reward -= 0.2 * (1 - torch.tanh(5 * tcp_to_obj_dist))

        # reach tcp to object reward
        goal_touch_pos = self.goal_cube.pose.p
        goal_touch_pos[..., 0] -= self.cube_half_sizes
        goal_touch_pos[..., -1] = 0
        tcp_to_obj_dist = torch.linalg.norm(
            (goal_touch_pos - tcp_pos),
            axis=-1,
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward += 2 * reaching_reward

        reward += info["touch_goal"].float()

        return reward - 0.1 * info["touch_table"]

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 3.4
