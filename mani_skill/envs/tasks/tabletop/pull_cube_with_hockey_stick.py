"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill tasks can minimally be defined by how the environment resets, what agents/objects are
loaded, goal parameterization, and success conditions

Environment reset is comprised of running two functions, `self._reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the positions of all objects (called actors), articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do
"""

from typing import Any, Dict, Union

import numpy as np
import torch
import sapien
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array





_stick_length = 0.2
_stick_end_length = 0.1
_stick_thickness = 5e-3 # y & z thicky
goal_thresh = 0.025

def _build_hockey_stick(
    scene: ManiSkillScene, 
    stick_length: float, 
    end_of_stick_length: float,
    stick_thickness: float,
):
    builder = scene.create_actor_builder()

    material = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5
    )

    half_sizes = [
        [stick_length, stick_thickness, stick_thickness], # long stick
        [stick_thickness, end_of_stick_length, stick_thickness], # end of stick
    ]

    poses = [
        sapien.Pose(p=[0, 0, 0]),
        sapien.Pose(p=[stick_length+ stick_thickness , end_of_stick_length - stick_thickness, 0]),
    ]
    for pose, half_size in zip(poses, half_sizes):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=material)

    return builder.build(name="hockey_stick")


@register_env("PullCubeWithHockeyStick-v1", max_episode_steps=100)
class PullCubeWithHockeyStickEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    # set some commonly used values
    goal_radius = 0.1
    cube_half_size = 0.02

    # same as pick_cube, stack_cube and push_cube
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)


    # same as pick_cube, stack_cube and push_cube
    @property 
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]


    # same as pick_cube, stack_cube and push_cube
    @property
    def _default_human_render_camera_configs(self):
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_scene(self, options: dict):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
        )

        self.hockey_stick = _build_hockey_stick(
            self.scene,
            stick_length=_stick_length,
            end_of_stick_length=_stick_end_length,
            stick_thickness=_stick_thickness,
        )

        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
        )

        # mock position, to check the end of stick pos and grasp stick pos are correct

        self.grasp_stick_region = actors.build_sphere(
            self.scene,
            radius=goal_thresh,
            color=[0, 1, 1, 1],
            name="grasp_stick_region",
            body_type="kinematic",
            add_collision=False,
        )

        self.long_stick_region = actors.build_sphere(
            self.scene,
            radius=goal_thresh,
            color=[0, 1, 0, 1],
            name="long_stick_region",
            body_type="kinematic",
            add_collision=False,
        )
        self._hidden_objects.append(self.long_stick_region)
        self._hidden_objects.append(self.grasp_stick_region)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # initial setup for handling multiple environments
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # set the cube's initial position
            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.cube_half_size
            q = [1, 0, 0, 0]
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.cube.set_pose(obj_pose)

            
            # set the goal's initial position
            target_region_xyz = xyz - torch.tensor([0.1 + self.goal_radius, 0, 0])
            # set a little bit above 0 so the target is sitting on the table
            target_region_xyz[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )


            # set the stick's initial position
            offset = torch.tensor([
                - (_stick_length - 2 * self.cube_half_size) , 
                - (_stick_end_length + 3* self.cube_half_size),
                0])
            target_region_xyz = xyz + offset
            target_region_xyz[..., 2] = _stick_thickness
            self.hockey_stick.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, 0, 0),
                )
            )

            # print("self.hockey_stick.pose.p:", self.hockey_stick.pose.p)
            # print("self.cube.pose.p:", self.cube.pose.p)

            # set long stick region to appear over the grasping region of the stick
            offset = torch.tensor([
                _stick_length + _stick_thickness,
                _stick_end_length,
                0])
            target_region_xyz = torch.tensor(self.hockey_stick.pose.p) + offset
            target_region_xyz[..., 2] = 4 * _stick_thickness
            self.long_stick_region.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, 0, 0),
                )
            )

            # set grasp stick region to appear over the end of the stick
            offset = torch.tensor([
                -_stick_length/2,
                0,
                0])
            target_region_xyz = torch.tensor(self.hockey_stick.pose.p)  + offset
            target_region_xyz[..., 2] = 4 * _stick_thickness
            self.grasp_stick_region.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, 0, 0),
                )
            )



    def _get_pos_of_end_of_stick(self):
        '''get the middle of end the shorter stick (end of the stick)'''
        offset = torch.tensor([
                _stick_length + _stick_thickness,
                _stick_end_length,
                0])
        return torch.tensor(self.hockey_stick.pose.p + offset).to(self.device)
    
    def _get_pos_of_grasp_stick(self):
        '''get the grasping position of the stick (3/4 of the stick length from the end of the stick)'''
        offset = torch.tensor([
                -_stick_length/2,
                0,
                0])
        return torch.tensor(self.hockey_stick.pose.p + offset).to(self.device)

    def _get_distances(self):
        '''
        return two distances:
        (1) from the end of the stick to the cube
        (2) from the robot to the grasp region
        '''
        # calcs for (1)
        dst_cube_to_end_of_stick = torch.linalg.norm(
            torch.tensor(self.cube.pose.p) - self._get_pos_of_end_of_stick(), axis=1
        )

        # calcs for (2)
        dst_robot_to_grasp_stick_pos = torch.linalg.norm(
            torch.tensor(self.agent.tcp.pose.p) - self._get_pos_of_grasp_stick(), axis=1
        )

        print("type(dst_cube_to_end_of_stick):", type(dst_cube_to_end_of_stick))
        print("type(dst_robot_to_grasp_stick_pos):", type(dst_robot_to_grasp_stick_pos))

        return dst_cube_to_end_of_stick, dst_robot_to_grasp_stick_pos



    def evaluate(self):
        a = self._get_pos_of_end_of_stick()


        is_obj_in_goal = (
            torch.linalg.norm(
                self.cube.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )
        is_grasped = self.agent.is_grasping(self.hockey_stick)
        is_robot_static = self.agent.is_static(0.2)

        # dst_cube_to_end_of_stick, dst_robot_to_grasp_stick_pos = self._get_distances()

        return {
            "success": is_obj_in_goal & is_robot_static,
            "is_obj_in_goal": is_obj_in_goal,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
            # "dst_cube_to_end_of_stick": dst_cube_to_end_of_stick,
            # "dst_robot_to_grasp_stick_pos": dst_robot_to_grasp_stick_pos,
        }

    def _get_obs_extra(self, info: Dict):
        # dst_cube_to_end_of_stick, dst_robot_to_grasp_stick_pos = self._get_distances()
        # default observartions
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose,)
        # if self._obs_mode in ["state", "state_dict"]:
        #     obs.update(
        #         obj_pose=self.cube.pose.raw_pose,
        #         dst_cube_to_end_of_stick = dst_cube_to_end_of_stick,
        #         dst_robot_to_grasp_stick_pos = dst_robot_to_grasp_stick_pos,
        #         stick_pose = self.hockey_stick.pose.raw_pose,
        #         obj_to_goal_dist = self.goal_region.pose.p - self.cube.pose.p,
        #     )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # dst_cube_to_end_of_stick, dst_robot_to_grasp_stick_pos = self._get_distances()

        # 1. Add reward the closer robot hand gets to the stick grasp pose
        # reaching_reward = 1 - torch.tanh(5 * dst_robot_to_grasp_stick_pos)
        # reward = reaching_reward


        # # 2. Add reward when we pick up the stick
        # is_grasped = info["is_grasped"]
        # reward+= is_grasped

        # # 3. Add reward as the distance of the end of the stick to the cube decreases
        # distance_reward = (1 - torch.tanh(5 * dst_cube_to_end_of_stick)) * is_grasped
        # reward += distance_reward

        # # 4. Add reward as the distance of the cube to the goal decreases
        # obj_to_goal_dist = torch.linalg.norm(
        #     self.goal_region.pose.p - self.cube.pose.p, axis=1
        # )
        # # place_reward = (1 - torch.tanh(5 * obj_to_goal_dist)) * is_grasped
        # place_reward = (1 - torch.tanh(5 * obj_to_goal_dist)) * is_grasped
        # reward += place_reward
        
        # # 4. Add reward when the robot is static
        # static_reward = 1 - torch.tanh(
        #     5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        # )
        # reward += static_reward * info["is_obj_in_goal"]

        # reward[info["success"]] = 5
        # # return reward
        # # convert reward to a tensor and return
        # print("type(reward):", type(reward))
        # return reward

        # temp return
        is_grasped = info["is_grasped"]
        reward = is_grasped
        return reward


    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
