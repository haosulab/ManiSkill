from typing import Any, Dict

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import PandaStick
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure GPU 0 is used for both sim and render
@register_env("RotateArrow-v1", max_episode_steps=50)
class RotateArrowEnv(BaseEnv):
    """
    **Task Description:**
    The goal is to pick up a book and place it inside a shelf with other books already in it.

    **Randomizations:**
    - books on the table have their z-axis rotation randomized.
    - books have their xy positions on top of the table scene randomized. The positions are sampled such that the books do not collide with each other.

    **Success Conditions:**
    - the book is inside the shelf. (to within half of the book size)
    - the book is static
    - the book is not being grasped by the robot (robot must let go of the cube)

    """
    # 3D T center of mass spawnbox dimensions
    arrow_spawnbox_xlength = 0.1
    arrow_spawnbox_ylength = 0.15

    # translation of the spawnbox from goal tee as upper left of spawnbox
    arrow_spawnbox_xoffset = -0.2
    arrow_spawnbox_yoffset = -0.05
    #  end randomizations - rotation around z is simply uniform

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: PandaStick

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        # sim_backend="physx_cuda:0", render_backend="sapien_cuda:0"
        if self.scene is not None:
            print(f"Is GPU simulation enabled for this scene? {self.scene.gpu_sim_enabled}")


    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.3, 0, 0.6], target=[-0.1, 0, -0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.6, -0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0])) # Loads the panda arm

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        # All values obtained carefully from blender
        # collision_boxes_shelf = [([0.43/2, 0.36/2, 0.05/2], sapien.Pose(p=[0,0.04, 0.015], q=[0.707, -0.707, 0, 0])),
        #                    ([0.025/2,0.25/2,0.235/2],sapien.Pose(p=[0.144,0.1729,0.023], q=[1,0,0,0])),
        #                    ([0.025/2, 0.25/2, 0.235/2], sapien.Pose(p=[-0.136, 0.1729, 0.023], q=[1,0,0,0])),
        #                    ([0.30/2,0.02/2,0.25/2],sapien.Pose(p=[0.0,0.161,-0.1],q=[0.707,0.707,0,0])),
        #                    ([0.312/2,0.27/2,0.0302/2],sapien.Pose(p=[0.0,0.286,0.018],q=[0.707,-0.707,0,0]))]
        self.arrow = self.load_glb_as_actor(self.scene, 
                                            "mani_skill/assets/push_arrow/arrow.glb", 
                                            sapien.Pose(p=[0.293, -0.1, 0], q=[-0.5, -0.5, 0.5, 0.5]), 
                                            name="arrow",
                                            type="dynamic")


    @staticmethod
    def load_glb_as_actor(scene, glb_file_path, pose, name, type="static"):
        """Load GLB file as a static actor in the scene"""
        builder = scene.create_actor_builder()
        builder.add_visual_from_file(glb_file_path)
        builder.add_multiple_convex_collisions_from_file(glb_file_path, decomposition="coacd")
        
        # for half_size, box_pose in collision_boxes:
        #     builder.add_box_collision(half_size=half_size, pose=box_pose)
        # try:
        #     # Some kind of error with shape over here.
        #     mesh_scene = trimesh.load(glb_file_path, force='scene')
        #     for geom_name, geometry in mesh_scene.geometry.items():
        #         print(geom_name)
        #         if geom_name.startswith("collision_"):
        #             # For each collision mesh, get its vertices and add a convex collision shape
        #             # The vertices are transformed to be relative to the object's origin
        #             vertices = geometry.vertices @ mesh_scene.graph.get(geom_name)[0].T
        #             builder.add_convex_collision_from_points(points=vertices)
        # except Exception as e:
        #     print(f"Warning: Failed to load collision mesh from {glb_file_path} with trimesh. Error: {e}")
        #     # Fallback to a single convex collision if trimesh fails or finds nothing
        #     builder.add_convex_collision_from_file(glb_file_path)
        # builder.add_nonconvex_collision_from_file(glb_file_path)
        builder.set_initial_pose(pose)
        if type=="dynamic":
            actor = builder.build_dynamic(name)
        else:
            actor = builder.build_static(name)
        return actor

    def quat_to_z_euler(self, quats):
        assert len(quats.shape) == 2 and quats.shape[-1] == 4
        # z rotation == can be defined by just qw = cos(alpha/2), so alpha = 2*cos^{-1}(qw)
        # for fixing quaternion double covering
        # for some reason, torch.sign() had bugs???
        signs = torch.ones_like(quats[:, -1])
        signs[quats[:, -1] < 0] = -1.0
        qw = quats[:, 0] * signs
        z_euler = 2 * qw.acos()
        return z_euler

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # setting the goal tee position, which is fixed, offset from center, and slightly rotated
            target_region_xyz = torch.zeros((b, 3))

#             # randomization code that randomizes the x, y position of the arrow we
#             # goal tee is alredy at y = -0.1 relative to robot, so we allow the tee to be only -0.2 y relative to robot arm
            target_region_xyz[..., 0] += (
                torch.rand(b) * (self.arrow_spawnbox_xlength) + self.arrow_spawnbox_xoffset
            )
            target_region_xyz[..., 1] += (
                torch.rand(b) * (self.arrow_spawnbox_ylength) + self.arrow_spawnbox_yoffset
            )

            target_region_xyz[..., 2] = (
                0.01982/2 + 2*1e-3
            )  # this is the half thickness of the tee plus a little

#             # rotation for pose is just random rotation around z axis
#             # z axis rotation euler to quaternion = [cos(theta/2),0,0,sin(theta/2)]

            q_euler_angle = torch.rand(b) * (2 * torch.pi)
            q = torch.zeros((b, 4))
            q = euler2quat(np.pi/2, 0, q_euler_angle)
            self.init_angle = q_euler_angle
            obj_pose = Pose.create_from_pq(p=target_region_xyz, q=q)
            self.arrow.set_pose(obj_pose)


    def evaluate(self):
        # pos_shelf = self.shelf.pose.p
        # pos_book = self.book_A.pose.p
        # offset = pos_shelf - pos_book
        # x_flag = torch.abs(offset[..., 0]) <= 0.13 + 0.005
        # y_flag = (
        #     torch.abs(offset[..., 1]) <= 0.18 + 0.005
        # )
        # z_flag = torch.abs(offset[..., 2]) <= 0.16 + 0.005
        # is_book_in_shelf = torch.logical_and(torch.logical_and(x_flag, y_flag),  z_flag)

        # # NOTE (stao): GPU sim can be fast but unstable. Angular velocity is rather high despite it not really rotating
        # is_book_static = self.book_A.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        # is_book_grasped = self.agent.is_grasping(self.book_A)
        # success = is_book_in_shelf * is_book_static * (~is_book_grasped)
        # return {
        #     "is_book_grasped": is_book_grasped,
        #     "is_book_in_shelf": is_book_in_shelf,
        #     "is_book_static": is_book_static,
        #     "success": success.bool()
        # }
        arrow_z_eulers = self.quat_to_z_euler(self.arrow.pose.q)
        rot_rew = (arrow_z_eulers - self.init_angle + torch.pi).cos()
        # print(f"Rotation reward: {rot_rew}")
        success = (rot_rew >= 0.9)
        # print(success)
        return {"success": success}

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            # state based gets info on goal position and t full pose - necessary to learn task
            obs.update(
                # goal_pos=self.goal_arrow.pose.p,
                obj_pose=self.arrow.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reward for overlap of the tees

        # legacy reward
        # reward = self.pseudo_render_reward()
        # Pose based reward below is preferred over legacy reward
        # legacy reward gets stuck in local maxs of 50-75% intersection
        # and then fails to promote large explorations to perfectly orient the T, for PPO algorithm

        # new pose based reward: cos(z_rot_euler) + function of translation, between target and goal both in [0,1]
        # z euler cosine similarity reward: -- quat_to_z_euler guarenteed to reutrn value from [0,2pi]
        arrow_z_eulers = self.quat_to_z_euler(self.arrow.pose.q)
        # subtract the goal z rotatation to get relative rotation
        rot_rew = (arrow_z_eulers - self.init_angle+torch.pi).cos()
        # cos output [-1,1], we want reward of mean 0.5
        reward = (rot_rew + 1) / 2
        # x and y distance as reward
        # arrow_to_goal_pose = self.arrow.pose.p[:, 0:2] - self.goal_arrow.pose.p[:, 0:2]
        # arrow_to_goal_pose_dist = torch.linalg.norm(arrow_to_goal_pose, axis=1)
        # reward += ((1 - torch.tanh(5 * arrow_to_goal_pose_dist)) ** 2) / 2

        # giving the robot a little help by rewarding it for having its end-effector close to the tee center of mass
        # tcp_to_push_pose = self.arrow.pose.p - self.agent.tcp.pose.p
        # tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        # reward += ((1 - torch.tanh(5 * tcp_to_push_pose_dist)).sqrt()) / 20

        # assign rewards to parallel environments that achieved success to the maximum of 3.
        reward[info["success"]] = 3
        return reward


    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 1
