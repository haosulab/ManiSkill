from typing import Any, Dict, Union
import numpy as np
import sapien
import torch
import trimesh
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from math import fabs
from mani_skill.utils.geometry import rotation_conversions

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure GPU 0 is used for both sim and render
@register_env("PlaceBookInShelf-v1", max_episode_steps=50)
class PlaceBookEnv(BaseEnv):
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

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

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
        self.shelf = self.load_glb_as_actor(self.scene, 
                                            "mani_skill/assets/book_in_shelf/BookShelf.glb", 
                                            sapien.Pose(p=[0.293, -0.1, 0], q=[-0.5, -0.5, 0.5, 0.5]), 
                                            name="custom_glb_shelf",
                                            type="static")
                                            
        
        self.book_A = self.load_glb_as_actor(self.scene, 
                                             "mani_skill/assets/book_in_shelf/simple_book_1.glb",
                                            sapien.Pose(p=[0.055, -0.158, 0.1], q=[0.854,0.471,0.212,0.068]),
                                            name="book_A",
                                            type="dynamic")
        # self.book_B = self.load_glb_as_actor(self.scene, 
        #                                      "mani_skill/assets/book_in_shelf/simple_book_2.glb",
        #                                     [([0.04,0.015,0.1], sapien.Pose(p=[0,0,0], q=[1,0,0,0]))],
        #                                     sapien.Pose(p=[0.0, -0.158, 0.1], q=[0.707,0,-0.707,0]),
        #                                     name="book_B")
        # self.cubeA = actors.build_cube(
        #     self.scene,
        #     half_size=0.02,
        #     color=[1, 0, 0, 1],
        #     name="cubeA",
        #     initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        # )
        # self.cubeB = actors.build_cube(
        #     self.scene,
        #     half_size=0.02,
        #     color=[0, 1, 0, 1],
        #     name="cubeB",
        #     initial_pose=sapien.Pose(p=[1, 0, 0.1]),
        # )


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
        print(f"{name} imported successfully")
        return actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.008
            # xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.2, -0.2], [0.0, 0.0]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            bookA_xy = sampler.sample(radius, 100)
            # cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = bookA_xy
            # qs = randomization.random_quaternions(
            #     b,
            #     lock_x=True,
            #     lock_y=True,
            #     lock_z=True,
            # )
            # [0.854,0.471,0.212,0.068] - q for sleeping book
            self.book_A.set_pose(Pose.create_from_pq(p=xyz.clone(), q=torch.tensor([0.748, 0.279, -0.464, 0.384]).repeat(b,1)))

            # xyz[:, :2] = cubeB_xy
            # qs = randomization.random_quaternions(
            #     b,
            #     lock_x=True,
            #     lock_y=True,
            #     lock_z=False,
            # )
            # self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def evaluate(self):
        pos_shelf = self.shelf.pose.p
        pos_book = self.book_A.pose.p
        offset = pos_shelf - pos_book
        x_flag = torch.abs(offset[..., 0]) <= 0.13 + 0.005
        y_flag = (
            torch.abs(offset[..., 1]) <= 0.18 + 0.005
        )
        z_flag = torch.abs(offset[..., 2]) <= 0.16 + 0.005
        is_book_in_shelf = torch.logical_and(torch.logical_and(x_flag, y_flag),  z_flag)

        # NOTE (stao): GPU sim can be fast but unstable. Angular velocity is rather high despite it not really rotating
        is_book_static = self.book_A.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_book_grasped = self.agent.is_grasping(self.book_A)
        success = is_book_in_shelf * is_book_static * (~is_book_grasped)
        return {
            "is_book_grasped": is_book_grasped,
            "is_book_in_shelf": is_book_in_shelf,
            "is_book_static": is_book_static,
            "success": success.bool()
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                shelf_pose=self.shelf.pose.raw_pose,
                book_pose=self.book_A.pose.raw_pose,
                tcp_to_shelf_pos=self.shelf.pose.p - self.agent.tcp.pose.p,
                tcp_to_book_pos=self.book_A.pose.p - self.agent.tcp.pose.p,
                book_to_shelf_pos=self.shelf.pose.p - self.book_A.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # rotation reward as cosine similarity between peg direction vectors
        # peg center of mass to end of peg, (1,0,0), rotated by peg pose rotation
        # dot product with its goal orientation: (0,0,1) or (0,0,-1)
        qmats = rotation_conversions.quaternion_to_matrix(self.book_A.pose.q)
        vec = torch.tensor([-1.0, 0, 0], device=self.device)
        goal_vec = torch.tensor([0, 0, 1.0], device=self.device)
        rot_vec = (qmats @ vec).view(-1, 3)
        # abs since (0,0,-1) is also valid, values in [0,1]
        rot_rew = (rot_vec @ goal_vec).view(-1).abs()
        reward = rot_rew

        # position reward using common maniskill distance reward pattern
        # giving reward in [0,1] for moving center of mass toward half length above table
        z_dist = torch.abs(self.book_A.pose.p[:, 2] - 0.16)
        reward += 1 - torch.tanh(5 * z_dist)

        # small reward to motivate initial reaching
        # initially, we want to reach and grip peg
        to_grip_vec = self.book_A.pose.p - self.agent.tcp.pose.p
        to_grip_dist = torch.linalg.norm(to_grip_vec, axis=1)
        reaching_rew = 1 - torch.tanh(5 * to_grip_dist)
        # reaching reward granted if gripping block
        reaching_rew[self.agent.is_grasping(self.book_A)] = 1
        # weight reaching reward less
        reaching_rew = reaching_rew / 5
        reward += reaching_rew

        reward[info["success"]] = 3

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8
