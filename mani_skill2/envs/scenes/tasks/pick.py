from .sequential_task import SequentialTaskEnv
from .planner import (
    TaskPlan,
    Subtask, PickSubtask,
    SubtaskConfig, PickSubtaskConfig,
)

import mani_skill2.envs.utils.randomization as randomization
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.structs.pose import Pose
from mani_skill2.utils.sapien_utils import compute_total_impulse, to_tensor
from mani_skill2.utils.geometry.rotation_conversions import quaternion_raw_multiply
import sapien
import sapien.physx as physx

import numpy as np
import torch
import torch.random

from tqdm import tqdm
from functools import cached_property
import itertools
from typing import Any, Dict, List, Tuple


PICK_OBS_EXTRA_KEYS = [
    "tcp_pose_wrt_base",
    "obj_pose_wrt_base",
    "is_grasped",
]


@register_env("PickSequentialTask-v0", max_episode_steps=200)
class PickSequentialTaskEnv(SequentialTaskEnv):
    """
    Task Description
    ----------------
    Add a task description here

    Randomizations
    --------------

    Success Conditions
    ------------------

    Visualization: link to a video/gif of the task being solved
    """

    # TODO (arth): add locomotion, open fridge, close fridge
    # TODO (arth) maybe?: clean this up, e.g. configs per subtask **type** or smth
    ee_rest_pos_wrt_base = Pose.create_from_pq(p=(0.5, 0, 1.25))
    pick_cfg = PickSubtaskConfig(
        horizon=200,
        ee_rest_thresh=0.05,
    )
    place_cfg = None

    def __init__(
            self,
            *args,
            robot_uids="fetch",
            task_plans: List[TaskPlan] = [],

            # spawn randomization
            randomize_arm=False,
            randomize_base=False,
            randomize_loc=False,

            # additional spawn randomization, shouldn't need to change
            spawn_loc_radius=2,

            # colliison tracking
            robot_force_mult=0,
            robot_force_penalty_min=0,
            robot_cumulative_force_limit=np.inf,

            **kwargs,
        ):

        # NOTE (arth): task plan length and order checking left to SequentialTaskEnv
        tp0 = task_plans[0]
        assert (
            len(tp0) == 1 and isinstance(tp0[0], PickSubtask),
            "Task plans for Pick training must be one PickSubtask long"
        )

        # randomization vals
        self.randomize_arm = randomize_arm
        self.randomize_base = randomize_base
        self.randomize_loc = randomize_loc
        self.spawn_loc_radius = spawn_loc_radius

        # force reward hparams
        self.robot_force_mult = robot_force_mult
        self.robot_force_penalty_min = robot_force_penalty_min
        self.robot_cumulative_force = 0
        self.robot_cumulative_force_limit = robot_cumulative_force_limit

        super().__init__(*args, robot_uids=robot_uids, task_plans=task_plans, **kwargs)


    # -------------------------------------------------------------------------------------------------
    # COLLISION TRACKING
    # -------------------------------------------------------------------------------------------------
    # TODO (arth): better version w/ new collision API
    # -------------------------------------------------------------------------------------------------

    def get_info(self):
        info = super().get_info()

        force = self._get_robot_force()
        info['robot_force'] = force

        self.robot_cumulative_force += force
        info['robot_cumulative_force'] = self.robot_cumulative_force

        return info
    
    def reset(self, *args, **kwargs):
        self.robot_cumulative_force = 0
        return super().reset(*args, **kwargs)
    
    def _get_actor_contacts(
        self, contacts: List[physx.PhysxContact], actor: sapien.Entity,
        ignore_collision_entities=set(),
    ) -> List[Tuple[physx.PhysxContact, bool]]:
        entity_contacts = []
        for contact in contacts:
            if (
                contact.bodies[0].entity == actor and 
                contact.bodies[1].entity not in ignore_collision_entities
            ):
                entity_contacts.append((contact, True))
            elif (
                contact.bodies[1].entity == actor and 
                contact.bodies[0].entity not in ignore_collision_entities
            ):
                entity_contacts.append((contact, False))
        return entity_contacts

    def _get_robot_force(self, ignore_grippers=True):
        contacts = self._scene.get_contacts()

        robot_impulse = 0
        for rle in self._robot_link_entities:
            if ignore_grippers and rle in self._robot_finger_entities:
                continue
            rle_impulse = compute_total_impulse(self._get_actor_contacts(
                contacts, rle, ignore_collision_entities=self.force_rew_ignore_entities,
            ))
            robot_impulse += np.linalg.norm(rle_impulse)
        
        return robot_impulse * self._sim_freq
    

    # -------------------------------------------------------------------------------------------------
    # INIT RANDOMIZATION
    # -------------------------------------------------------------------------------------------------
    # TODO (arth): integrate with navigable base position thing once that's done
    #       also maybe check that obj won't fall when noise is added
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    
    def _get_navigable_spawn_positions_with_rots_and_dists(self, center_x, center_y):
        # NOTE (arth): this is all unbatched, should be called wtih DEFAULT obj spawn pos
        center = torch.tensor([center_x, center_y])
        pts = torch.tensor(self.scene_builder.navigable_positions)
        pts_wrt_center = pts - center

        dists = torch.norm(pts_wrt_center, dim=1)
        in_circle = dists <= self.spawn_loc_radius
        pts, pts_wrt_center, dists = pts[in_circle], pts_wrt_center[in_circle], dists[in_circle]

        rots = torch.sign(pts_wrt_center[:, 1]) * torch.arccos(pts_wrt_center[:, 0] / dists) + torch.pi
        rots %= 2 * torch.pi

        return torch.hstack([pts, rots.unsqueeze(-1)]), dists


    def reconfigure(self):
        # run reconfiguration
        super().reconfigure()

        # links and entities for force tracking
        self._obj_entity = self.subtask_objs[0]._bodies[0].entity
        self._robot_link_entities = [
            x._bodies[0].entity for x in self.agent.robot.get_links()
        ]
        self._robot_finger_entities = set([
            x._bodies[0].entity for x in [self.agent.finger1_link, self.agent.finger2_link]
        ])
        self.force_rew_ignore_entities = set(self._robot_link_entities + [self._obj_entity])


        self.scene_builder.set_actor_default_poses_vels()

        # NOTE (arth): targ obj should be same across gpu envs
        #   and same default pose, so we use unbatched pose
        obj = self.subtask_objs[0]
        center = obj.pose.p[0, :2]

        self.spawn_loc_rot, dists = self._get_navigable_spawn_positions_with_rots_and_dists(
            center[0], center[1]
        )
        
        # TODO: (arth) implement with gpu sim?
        accept_spawn_loc_rot = []
        qpos = self.agent.RESTING_QPOS
        self.scene_builder.disable_fetch_ground_collisions()

        for x, y, z_rot in tqdm(self.spawn_loc_rot, total=len(self.spawn_loc_rot)):
            robot_force = 0
            for delta_x, delta_y in itertools.product([0.1, -0.1], [0.1, -0.1]):
                self.agent.controller.reset()
                qpos[..., 2] = z_rot
                self.agent.reset(qpos)
                self.agent.robot.set_pose(sapien.Pose(p=[
                    x + delta_x, y + delta_y, self.agent.robot.pose.p[0, 2]
                ]))
                self._scene.step()
                robot_force += self._get_robot_force()
            accept_spawn_loc_rot.append(robot_force <= 1e-3)
        self.spawn_loc_rot = self.spawn_loc_rot[accept_spawn_loc_rot]

        dists = dists[accept_spawn_loc_rot]
        self.closest_spawn_loc_rot = self.spawn_loc_rot[torch.argmin(dists)]


    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    

    def _initialize_actors(self, env_idx):
        with torch.device(self.device):
            super()._initialize_actors(env_idx)
            b = len(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.12
            xyz += self.subtask_objs[0].pose.p

            qs = quaternion_raw_multiply(
                randomization.random_quaternions(
                    b, lock_x=True, lock_y=True, lock_z=False
                ),
                self.subtask_objs[0].pose.q,
            )
            self.subtask_objs[0].set_pose(Pose.create_from_pq(xyz, qs))

    # TODO (arth): figure out rejection pipeline including arm/base randomization
    #       tbh not sure how to do yet, might just increase collision thresholds in training
    def _initialize_agent(self, env_idx):
        with torch.device(self.device):
            b = len(env_idx)

            # NOTE (arth): it is assumed that scene builder spawns agent with some qpos
            qpos = self.agent.robot.get_qpos()
            if self.randomize_loc:
                loc_rot = self.spawn_loc_rot[torch.randint(
                    high=len(self.spawn_loc_rot), size=(b,)
                )]
            else:
                loc_rot = self.closest_spawn_loc_rot.unsqueeze(0).repeat(b, 1)
            
            robot_pos = self.agent.robot.pose.p
            robot_pos[..., :2] = loc_rot[..., :2]
            self.agent.robot.set_pose(Pose.create_from_pq(p=robot_pos))
            qpos[..., 2] = loc_rot[..., 2]

            if self.randomize_base:
                # base pos
                robot_pos = self.agent.robot.pose.p
                robot_pos[..., :2] += torch.clip(torch.normal(
                    0, 0.1, (b, len(robot_pos[0, :2]))
                ), -0.1, 0.1)
                self.agent.robot.set_pose(Pose.create_from_pq(p=robot_pos))
                # base rot
                qpos[..., 2:3] += torch.clip(torch.normal(
                    0, 0.25, (b, len(qpos[0, 2:3]))
                ), -0.5, 0.5)
            if self.randomize_arm:
                qpos[..., 5:6] += torch.clip(torch.normal(
                    0, 0.05, (b, len(qpos[0, 5:6]))
                ), -0.1, 0.1)
                qpos[..., 7:-2] += torch.clip(torch.normal(
                    0, 0.05, (b, len(qpos[0, 7:-2]))
                ), -0.1, 0.1)
            self.agent.reset(qpos)

    # -------------------------------------------------------------------------------------------------


    # -------------------------------------------------------------------------------------------------
    # OBS AND INFO
    # -------------------------------------------------------------------------------------------------
    # Remove irrelevant obs for pick task from state dict
    # -------------------------------------------------------------------------------------------------
    
    def _get_obs_state_dict(self, info: Dict):
        state_dict = super()._get_obs_state_dict(info)

        extra_state_dict_keys = list(state_dict["extra"])
        for key in extra_state_dict_keys:
            if key not in PICK_OBS_EXTRA_KEYS:
                state_dict["extra"].pop(key, None)

        return state_dict
    
    # -------------------------------------------------------------------------------------------------


    # -------------------------------------------------------------------------------------------------
    # REWARD
    # -------------------------------------------------------------------------------------------------
    # NOTE (arth): evaluate() function here to support continuous task wrapper on cpu sim
    # -------------------------------------------------------------------------------------------------

    def evaluate(self):
        infos = super().evaluate()

        # set to zero in case we use continuous task wrapper in cpu sim
        #   this way, if the termination signal is ignored, env will
        #   still reevalate success each step
        self.subtask_pointer = torch.zeros_like(self.subtask_pointer)

        return infos

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        with torch.device(self.device):
            reward = torch.zeros(self.num_envs)

            obj_pos = self.subtask_objs[0].pose.p
            goal_pos = self.ee_rest_world_pose.p
            tcp_pos = self.agent.tcp_pose.p

            robot_to_obj_dist = torch.norm(
                self.agent.torso_lift_link.pose.p - self.agent.tcp_pose.p, dim=1
            )


            # NOTE (arth): reward steps are as follows:
            #       - if too far fom object:
            #           - move_to_obj_reward
            #       - else
            #           - reaching_reward
            #           - if not grasped
            #               - not_grasped_reward
            #           - is_grasped_reward
            #           - if grasped
            #               - grasped_rewards
            #           - if grasped and ee_at_rest
            #               - static_reward
            #           - success_reward
            # ---------------------------------------------------
            # CONDITION CHECKERS
            # ---------------------------------------------------

            robot_too_far = robot_to_obj_dist > self.agent.REACHABLE_DIST
            too_far_reward = torch.zeros_like(reward[robot_too_far])

            robot_close_enough = ~robot_too_far
            close_enough_reward = torch.zeros_like(reward[robot_close_enough])

            not_grasped = robot_close_enough & ~info['is_grasped']
            not_grasped_reward = torch.zeros_like(reward[not_grasped])

            is_grasped = robot_close_enough & info['is_grasped']
            is_grasped_reward = torch.zeros_like(reward[is_grasped])

            ee_rest = (
                robot_close_enough
                & is_grasped
                & (torch.norm(tcp_pos - goal_pos, dim=1) <= self.pick_cfg.ee_rest_thresh)
            )
            ee_rest_reward = torch.zeros_like(reward[ee_rest])

            # ---------------------------------------------------


            reaching_reward, grasp_reward, success_reward, torso_move_pen, \
                ee_jitter_pen, ee_not_over_obj_pen, place_reward, base_move_rot_pen, \
                torso_move_down_pen, still_on_table_pen, arm_resting_orientation_pen, static_reward \
                = tuple([0] * 12)


            if torch.any(robot_too_far):
                # prevent torso and arm moving too much
                arm_torso_qvel = self.agent.robot.qvel[..., 3:-2][robot_too_far]
                arm_torso_move_pen = 2 * torch.tanh(
                    torch.norm(arm_torso_qvel, dim=1) / 5
                )
                too_far_reward -= arm_torso_move_pen

                # encourage robot to move closer to obj
                robot_far_pen = torch.tanh(robot_to_obj_dist[robot_too_far] / 5)
                too_far_reward -= robot_far_pen


            if torch.any(robot_close_enough):
                # reaching reward
                tcp_to_obj_pos = obj_pos[robot_close_enough] - tcp_pos[robot_close_enough]
                tcp_to_obj_dist = torch.norm(tcp_to_obj_pos, dim=1)
                reaching_reward = (1 - torch.tanh(5 * tcp_to_obj_dist))
                close_enough_reward += reaching_reward

                # penalty for ee moving too much when not grasping
                ee_vel = self.agent.tcp.linear_velocity[robot_close_enough]
                ee_jitter_pen = torch.tanh(torch.norm(ee_vel, dim=1) / 5)
                close_enough_reward -= ee_jitter_pen

                # pick reward
                grasp_reward = 2 * info['is_grasped'][robot_close_enough]
                close_enough_reward += grasp_reward

                # success reward
                success_reward = 3 * info['success'][robot_close_enough]
                close_enough_reward += success_reward


            if torch.any(not_grasped):
                # penalty for torso moving up and down too much
                tqvel_z = self.agent.robot.qvel[..., 3][not_grasped]
                torso_move_pen = torch.tanh(5 * torch.abs(tqvel_z))
                not_grasped_reward -= torso_move_pen

                # penalty for ee not over obj
                ee_not_over_obj_pen = torch.tanh(5 * torch.norm(
                    obj_pos[..., :2][not_grasped] - tcp_pos[..., :2][not_grasped], dim=1
                ))
                not_grasped_reward -= ee_not_over_obj_pen


            if torch.any(is_grasped):
                # place reward
                ee_to_rest_dist = torch.norm(tcp_pos[is_grasped] - goal_pos[is_grasped], dim=1)
                place_reward = 4 * (1 - torch.tanh(3 * ee_to_rest_dist))
                is_grasped_reward += place_reward

                # penalty for torso moving down too much
                tqvel_z = torch.clip(self.agent.robot.qvel[..., 3][is_grasped], max=0)
                torso_move_down_pen = torch.tanh(5 * torch.abs(tqvel_z))
                is_grasped_reward -= torso_move_down_pen

                # penalty for base moving or rotating too much
                bqvel = self.agent.robot.qvel[..., :3][is_grasped]
                base_move_rot_pen = 1.5 * torch.tanh(torch.norm(bqvel, dim=1))
                is_grasped_reward -= base_move_rot_pen

                # encourage arm and torso in "resting" orientation
                arm_to_resting_diff = torch.norm(
                    self.agent.robot.qpos[..., 3:-2][is_grasped] - self.agent.RESTING_QPOS[3:-2], dim=1
                )
                arm_resting_orientation_pen = torch.tanh(arm_to_resting_diff / 5)
                is_grasped_reward -= arm_resting_orientation_pen


            if torch.any(ee_rest):
                qvel = self.agent.robot.qvel[..., :-2][ee_rest]
                static_reward = (1 - torch.tanh(torch.norm(qvel, dim=1)))
                ee_rest_reward += static_reward


            # add rewards to specific envs
            reward[robot_too_far] += too_far_reward
            reward[robot_close_enough] += close_enough_reward
            reward[not_grasped] += not_grasped_reward
            reward[is_grasped] += is_grasped_reward


            # step collision penalty
            step_col_pen = max(self.robot_force_mult * info['robot_force'], self.robot_force_penalty_min)
            reward -= torch.tensor([step_col_pen])

            # cumulative collision penatly
            cum_col_pen = float(info['robot_cumulative_force'] > self.robot_cumulative_force_limit)
            reward -= torch.tensor([cum_col_pen])

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 5.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
    
    # -------------------------------------------------------------------------------------------------

