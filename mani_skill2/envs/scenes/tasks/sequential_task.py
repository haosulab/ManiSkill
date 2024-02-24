from .planner import (
    TaskPlan,
    Subtask, PickSubtask, PlaceSubtask,
    SubtaskConfig, PickSubtaskConfig, PlaceSubtaskConfig,
)

from mani_skill2.envs.scenes.base_env import SceneManipulationEnv
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.utils.structs.actor import Actor
from mani_skill2.utils.structs.pose import Pose, vectorize_pose
from mani_skill2.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from mani_skill2.utils.building.actors import build_sphere
from mani_skill2.agents.robots import Fetch
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.visualization.misc import observations_to_images, tile_images
import sapien
import sapien.physx as physx

import numpy as np
import torch
import torch.random

from collections import OrderedDict
from typing import Any, Dict, List, Tuple

def all_equal(array: list):
    return len(set(array)) == 1
def all_same_type(array: list):
    return len(set(map(type, array))) == 1

@register_env("SequentialTask-v0")
class SequentialTaskEnv(SceneManipulationEnv):
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

    SUPPORTED_ROBOTS = ["fetch"]
    agent: Fetch
    sim_cfg = SimConfig(
        gpu_memory_cfg=GPUMemoryConfig(
            found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
        )
    )

    # TODO (arth): add locomotion, open fridge, close fridge
    # TODO (arth) maybe?: clean this up, e.g. configs per subtask **type** or smth
    EE_REST_POS_WRT_BASE = (0.5, 0, 1.25)
    pick_cfg = PickSubtaskConfig(
        horizon=200,
        ee_rest_thresh=0.05,
    )
    place_cfg = PlaceSubtaskConfig(
        horizon=200,
        obj_goal_thresh=0.015,
        ee_rest_thresh=0.05,
    )
    task_cfgs: Dict[str, SubtaskConfig] = dict(
        pick=pick_cfg,
        place=place_cfg,
    )

    def __init__(
            self,
            *args,
            robot_uids="fetch",
            robot_init_qpos_noise=0.02,
            task_plans: List[TaskPlan] = [],
            **kwargs,
        ):

        if 'num_envs' in kwargs:
            assert (
                len(task_plans) == kwargs['num_envs'],
                f"GPU sim requires equal number of task_plans ({len(task_plans)}) and parallel_envs ({kwargs['num_envs']})"
            )
        else:
            assert len(task_plans) == 1, f"CPU sim only supports one task plan, not {(len(task_plans))}"

        self.base_task_plans = task_plans
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # TODO (arth): setting subtask.obj causes a pickling error
        #   this is (imo) an ugly workaround so i'll try to figure smth else out
        self.subtask_objs: List[Actor] = []
        self.subtask_goals: List[Actor] = []
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    # -------------------------------------------------------------------------------------------------
    # PROCESS TASKS
    # -------------------------------------------------------------------------------------------------
    # NOTE (arth): since M3 only has finite defined tasks, we hardcode task init.
    # TODO (arth): maybe make this automatic with more general class structure? not sure yet
    # -------------------------------------------------------------------------------------------------

    def process_task_plan(self):

        assert (
            all_equal([len(plan) for plan in self.base_task_plans]),
            "All parallel task plans must be the same length"
        )
        assert(
            np.all([
                all_same_type(parallel_subtasks)
                for parallel_subtasks in zip(*self.base_task_plans)
            ]),
            "All parallel task plans must have same subtask types in same order"
        )

        # build new merged task_plan and merge actors of parallel task plants
        self.task_plan = []
        for subtask_num, parallel_subtasks in enumerate(zip(*self.base_task_plans)):
            subtask0 = parallel_subtasks[0]

            if isinstance(subtask0, PickSubtask):
                parallel_subtasks: List[PickSubtask]
                merged_obj_name = f"obj_{subtask_num}"

                self.subtask_objs.append(Actor.merge([
                    self._get_actor(subtask.obj_id) for subtask in parallel_subtasks
                ], name=merged_obj_name))
                self.subtask_goals.append(None)

                self.task_plan.append(PickSubtask(obj_id=merged_obj_name))
            
            elif isinstance(subtask0, PlaceSubtask):
                parallel_subtasks: List[PlaceSubtask]
                merged_obj_name = f"obj_{subtask_num}"
                merged_goal_name = f"goal_{subtask_num}"

                self.subtask_objs.append(Actor.merge([
                    self._get_actor(subtask.obj_id) for subtask in parallel_subtasks
                ], name=merged_obj_name))
                self.subtask_goals.append(Actor.merge([
                    self._make_goal(
                        pos=subtask.goal_pos,
                        radius=self.place_cfg.obj_goal_thresh,
                        name=f"goal_{subtask.uid}",
                    )
                    for subtask in parallel_subtasks
                ], name=merged_goal_name))

                self.task_plan.append(PlaceSubtask(
                    obj_id=merged_obj_name,
                    goal_pos=[subtask.goal_pos for subtask in parallel_subtasks]
                ))
                
            else:
                raise AttributeError(f"{subtask0.type} {type(subtask0)} not yet supported")
            
        self.task_horizons = torch.tensor([
            self.task_cfgs[subtask.type].horizon for subtask in self.task_plan
        ], device=self.device, dtype=torch.long)
        self.task_ids = torch.tensor([
            self.task_cfgs[subtask.type].task_id for subtask in self.task_plan
        ], device=self.device, dtype=torch.long)

        # TODO (arth): figure out how to change horizon after task inited
        # self.max_episode_steps = torch.sum(self.task_horizons)

    def _get_actor(self, actor_id: str):
        return self.scene_builder.movable_objects_by_id[actor_id]

    def _make_goal(
            self, pos: Tuple[float, float, float] = None, radius=0.15, name="goal_site",
        ):
        goal = build_sphere(
            self._scene,
            radius=radius,
            color=[0, 1, 0, 1],
            name=name,
            body_type="kinematic",
            add_collision=False,
        )
        if pos is not None:
            goal.set_pose(sapien.Pose(p=pos))
        self._hidden_objects.append(goal)
        return goal
    
    def _compute_ee_rest_world_pose(self):
        return self.agent.base_link.pose * self.ee_rest_pos_wrt_base
    
    # -------------------------------------------------------------------------------------------------
    

    # -------------------------------------------------------------------------------------------------
    # RESET/RECONFIGURE HANDLING
    # -------------------------------------------------------------------------------------------------

    def _load_actors(self):
        super()._load_actors()
        self.process_task_plan()
        self.subtask_pointer = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.subtask_steps_left = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.ee_rest_goal = self._make_goal(
            radius=0.05,
            name="ee_rest_goal",
        )
    
    def _initialize_task(self, env_idx: torch.Tensor):
        # TODO (arth): currently there's a bug where prev contacts/etc will maintain themselves somehow
        #       maybe bug will be fixed alter, but in meantime just step scene to get rid of old contacts
        if not sapien.physx.is_gpu_enabled():
            self.agent.controller.reset()
            self._scene.step()

        self.subtask_pointer[env_idx] = 0
        self.subtask_steps_left[env_idx] = self.task_cfgs[self.task_plan[0].type].horizon
        self.ee_rest_pos_wrt_base = Pose.create_from_pq(p=self.EE_REST_POS_WRT_BASE)
        self.ee_rest_world_pose = self._compute_ee_rest_world_pose()
    
    # -------------------------------------------------------------------------------------------------
    

    # -------------------------------------------------------------------------------------------------
    # SUBTASK STATUS CHECKERS/UPDATERS
    # -------------------------------------------------------------------------------------------------

    def evaluate(self):
        self.ee_rest_world_pose = self._compute_ee_rest_world_pose()
        subtask_success, is_grasped = self._subtask_check_success()

        self.subtask_pointer[subtask_success] += 1
        success = self.subtask_pointer >= len(self.task_plan)

        self.subtask_steps_left -= 1
        update_subtask_horizon = subtask_success & ~success
        self.subtask_steps_left[update_subtask_horizon] = self.task_horizons[
            self.subtask_pointer[update_subtask_horizon]
        ]

        fail = (self.subtask_steps_left <= 0) & ~success

        subtask_type = torch.full_like(self.subtask_pointer, len(self.task_plan))
        subtask_type[~success] = self.task_ids[self.subtask_pointer[[~success]]]

        return dict(
            success=success,
            fail=fail,
            is_grasped=is_grasped,
            subtask=self.subtask_pointer,
            subtask_type=subtask_type,
            subtasks_steps_left=self.subtask_steps_left,
        )

    # NOTE (arth): for now group by specific subtask (good enough for training)
    # TODO (arth): maybe group by relevant commonalities for batched computation to speed up?
    #       e.g. pick and place both use is_grasped computations, and concurrent envs might be using the
    #       same objs for their current subtasks, etc
    #       not sure yet, TBD
    def _subtask_check_success(self):
        subtask_success = torch.zeros(self.num_envs, device=self.device, dtype=bool)
        is_grasped = torch.zeros(self.num_envs, device=self.device, dtype=bool)

        currently_running_subtasks = torch.unique(
            torch.clip(self.subtask_pointer, max=len(self.task_plan) - 1)
        )
        for subtask_num in currently_running_subtasks:
            subtask: Subtask = self.task_plan[subtask_num]
            env_idx = torch.where(self.subtask_pointer == subtask_num)[0]
            if isinstance(subtask, PickSubtask):
                subtask_success[env_idx], is_grasped[env_idx] = self._pick_check_success(
                    self.subtask_objs[subtask_num], env_idx,
                    ee_rest_thresh=self.pick_cfg.ee_rest_thresh
                )
            elif isinstance(subtask, PlaceSubtask):
                subtask_success[env_idx], is_grasped[env_idx] = self._place_check_success(
                    self.subtask_objs[subtask_num], self.subtask_goals[subtask_num], env_idx,
                    obj_goal_thresh=self.place_cfg.obj_goal_thresh,
                    ee_rest_thresh=self.place_cfg.ee_rest_thresh,
                )
            else:
                raise AttributeError(f"{subtask.type} {type(subtask)} not supported")
        
        return subtask_success, is_grasped
                
    def _pick_check_success(
            self,
            obj: Actor, env_idx: torch.Tensor,
            ee_rest_thresh: float = 0.05,
        ):
        is_grasped = self.agent.is_grasping(obj, max_angle=30)[env_idx]
        ee_rest = torch.norm(
            self.agent.tcp_pose.p[env_idx] - self.ee_rest_world_pose.p[env_idx], dim=1
        ) <= ee_rest_thresh
        is_static = self.agent.is_static(threshold=0.2)[env_idx]
        return is_grasped & ee_rest & is_static, is_grasped
    
    def _place_check_success(
            self,
            obj: Actor, obj_goal: Actor, env_idx: torch.Tensor,
            obj_goal_thresh: float = 0.15, ee_rest_thresh: float = 0.05,
        ):
        is_grasped = self.agent.is_grasping(obj, max_angle=30)[env_idx]
        obj_at_goal = torch.linalg.norm(
            obj.pose.p[env_idx] - obj_goal.pose.p[env_idx], axis=1
        ) <= obj_goal_thresh
        ee_rest = torch.norm(
            self.agent.tcp_pose.p[env_idx] - self.ee_rest_world_pose.p[env_idx], dim=1
        ) <= ee_rest_thresh
        is_static = self.agent.is_static(threshold=0.2)[env_idx]
        return is_grasped & obj_at_goal & ee_rest & is_static, is_grasped

    # -------------------------------------------------------------------------------------------------


    # -------------------------------------------------------------------------------------------------
    # OBS AND INFO
    # -------------------------------------------------------------------------------------------------
    # NOTE (arth): 
    #       1. fetch hacked base needs x, y, z_rot qpos and qvel masked
    #       2. all tasks except locomotion use
    #           - depth images (not implemented yet)
    #           - arm joints
    #           - ee pos in base frame
    #           - gripper holding anything
    #           - target pos in base frame
    #               - e.g. pick target pos, place target pos, fridge handle pos, etc
    #       3. locomotion (not implemented yet)
    #           - only uses depth image and goal pos
    # -------------------------------------------------------------------------------------------------
    
    def _get_obs_agent(self):
        agent_state = super()._get_obs_agent()
        agent_state["qpos"][..., :3] = 0
        agent_state["qvel"][..., :3] = 0
        return agent_state
    

    # TODO (arth): maybe find better way of doing thing
    # NOTE (arth): for now, define keys that will always be added to obs. leave it to
    #       wrappers or task-specific envs to mask out unnecessary vals
    #       - subtasks that don't need that obs will set some default value
    #       - subtasks which need that obs will set value depending on subtask params
    def _get_obs_extra(self, info: Dict):
        base_pose_inv = self.agent.base_link.pose.inv()

        # all subtasks will have same computation for
        #       - tcp_pose_wrt_base :   tcp always there and is same link
        tcp_pose_wrt_base = vectorize_pose(base_pose_inv * self.agent.tcp.pose)

        #       - obj_pose_wrt_base :   different objs per subtask (or no obj)
        #       - goal_pos_wrt_base :   different goals per subtask (or no goal)
        obj_pose_wrt_base = torch.zeros(self.num_envs, 7, device=self.device, dtype=torch.float)
        goal_pos_wrt_base = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)

        currently_running_subtasks = torch.unique(
            torch.clip(self.subtask_pointer, max=len(self.task_plan) - 1)
        )
        for subtask_num in currently_running_subtasks:
            subtask: Subtask = self.task_plan[subtask_num]
            env_idx = torch.where(self.subtask_pointer == subtask_num)[0]
            if isinstance(subtask, PickSubtask):
                obj_pose_wrt_base[env_idx] = vectorize_pose(base_pose_inv * self.subtask_objs[subtask_num].pose)
            elif isinstance(subtask, PlaceSubtask):
                obj_pose_wrt_base[env_idx] = vectorize_pose(base_pose_inv * self.subtask_objs[subtask_num].pose)
                goal_pos_wrt_base[env_idx] = (base_pose_inv * self.subtask_goals[subtask_num].pose).p
            else:
                raise AttributeError(f"{subtask.type} {type(subtask)} not supported")

        # already computed during evaluation is
        #       - is_grasped    :   part of success criteria (or set default)
        is_grasped = info["is_grasped"]
        
        return OrderedDict(
            tcp_pose_wrt_base=tcp_pose_wrt_base,
            obj_pose_wrt_base=obj_pose_wrt_base,
            goal_pos_wrt_base=goal_pos_wrt_base,
            is_grasped=is_grasped,
        )

    # -------------------------------------------------------------------------------------------------


    # -------------------------------------------------------------------------------------------------
    # REWARD (Ignored here)
    # -------------------------------------------------------------------------------------------------
    # NOTE (arth): this env does not have dense rewards since rewards are used for training subtasks.
    #       If need to train a subtask, extend this class to define a subtask
    # -------------------------------------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.subtask_pointer

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
    
    # -------------------------------------------------------------------------------------------------


    # -------------------------------------------------------------------------------------------------
    # CAMERAS, SENSORS, AND RENDERING
    # -------------------------------------------------------------------------------------------------
    # NOTE (arth): also included the old "cameras" mode from MS2 since the robot render camera
    #       can get stuck in walls
    # -------------------------------------------------------------------------------------------------
    
    def _register_sensors(self):
        # for fetch, the only sensor used is the fetch head camera
        return []

    def _register_human_render_cameras(self):
        # # top-down camera (good for spawn generation vids)
        # room_camera_pose = look_at([0, 0, 12], [0, 0, 0])
        # room_camera_config = CameraConfig(
        #     "render_camera", room_camera_pose.p, room_camera_pose.q, 512, 512, 1, 0.01, 20
        # )
        # return room_camera_config
    
        # this camera follows the robot around (though might be in walls if the space is cramped)
        robot_camera_pose = look_at([2, 0, 1], [0, 0, 0])
        robot_camera_config = CameraConfig(
            "render_camera", robot_camera_pose.p, robot_camera_pose.q, 512, 512, 1.5, 0.01, 10,
            link=self.agent.torso_lift_link
        )
        return robot_camera_config
    
    def render_cameras(self):
        images = []
        for obj in self._hidden_objects:
            obj.hide_visual()
        self._scene.update_render()
        self.capture_sensor_data()
        sensor_images = self.get_sensor_obs()
        for sensor_images in sensor_images.values():
            images.extend(observations_to_images(sensor_images))
        return tile_images([self.render_rgb_array()] + images)
    
    def render_rgb_array(self):
        self.ee_rest_goal.set_pose(self.ee_rest_world_pose)
        return super().render_rgb_array()

    def render_human(self):
        self.ee_rest_goal.set_pose(self.ee_rest_world_pose)
        return super().render_human()

    def render(self):
        if self.render_mode == "cameras":
            return self.render_cameras()
        
        return super().render()
    
    # -------------------------------------------------------------------------------------------------
