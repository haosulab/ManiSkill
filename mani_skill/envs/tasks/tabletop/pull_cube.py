from typing import Any, Dict, Union
import numpy as np
import torch
import sapien
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("PullCubeTool-v1", max_episode_steps=50)
class PullCubeToolEnv(BaseEnv):
    """
    Task Description
    -----------------
    Given an L-shaped tool that is within the reach of the robot, leverage the
    tool to pull a cube that is out of it's reach

    Randomizations
    ---------------
    - The cube's position (x,y) is randomized on top of a table in the region "<out of manipulator
    reach, but within reach of tool>". It is placed flat on the table
    - The target goal region is the region on top of the table marked by "<within reach of arm>"

    Success Conditions
    -----------------
    - The cube's xy position is within the goal region of the arm's base (marked by reachability)
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse", "none")
    agent: Union[Panda, Fetch]

    goal_radius = 0.1
    cube_half_size = 0.01
    handle_length = 0.20
    hook_length = 0.05
    width = 0.02
    height = 0.02
    cube_size = 0.02
    arm_reach = 0.35

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # create cube
        self.obj = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
        )

        # create target
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.cube_half_size
            q = [1, 0, 0, 0]

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

            target_region_xyz = xyz - torch.tensor([0.1 + self.goal_radius, 0, 0])

            target_region_xyz[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(
                self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )

        return {
            "success": is_obj_placed,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_region.pose.p,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.cube.pose.p
        tool_pos = self.l_shape_tool.pose.p
        
        
        tool_grasp_pos = tool_pos + torch.tensor(
            [0.02, 0, 0],  # Slight offset for better grasping
            device=self.device
        )
        tcp_to_tool_dist = torch.linalg.norm(tcp_pos - tool_grasp_pos, dim=1)
        reaching_tool_reward = 1 - torch.tanh(5.0 * tcp_to_tool_dist)
        
        
        tool_reached = tcp_to_tool_dist < 0.01
        
        
        hook_end_pos = tool_pos + torch.tensor(
            [self.handle_length - self.hook_length/2, self.width, 0], 
            device=self.device
        )
        ideal_hook_pos = cube_pos + torch.tensor(
            [-self.cube_half_size - 0.02, 0, 0],  
            device=self.device
        )
        hook_to_target_dist = torch.linalg.norm(hook_end_pos - ideal_hook_pos, dim=1)
        tool_positioning_reward = 1 - torch.tanh(5.0 * hook_to_target_dist)
        
        
        workspace_center = torch.zeros((len(cube_pos), 3), device=self.device)
        workspace_center[:, 0] = self.arm_reach * 0.7
        cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_center, dim=1)
        cube_progress_reward = 1 - torch.tanh(5.0 * cube_to_workspace_dist)
        
        
        reward = reaching_tool_reward
        reward += tool_positioning_reward * tool_reached
        reward += cube_progress_reward * (hook_to_target_dist < 0.02)  # Only when hook is positioned
        
        
        if "success" in info:
            reward[info["success"]] = 3.0
            
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        # CHANGE: Use same max reward as template
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward