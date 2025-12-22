from typing import Any, Dict, Union
import numpy as np
import sapien
import torch
import trimesh
import os
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.scene_builder.robocasa.fixtures.cabinet import OpenCabinet
from mani_skill.utils.structs.pose import Pose
from math import fabs
from mani_skill.utils.geometry import rotation_conversions

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure GPU 0 is used for both sim and render
@register_env("PickSodaFromCabinet-v1", max_episode_steps=50)
class PickSodaFromCabinetEnv(BaseEnv):
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
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0])) # Loads the panda arm

    def _load_scene(self, options: dict):
        # Check if RoboCasa dataset is available
        self._check_robocasa_dataset()
        
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # self.cabinet_scene = RoboCasaSceneBuilder(
        #     env=self, init_robot_base_pos=sapien.Pose(p=[4, -0.6, 0.94], q=[ 0.7071, 0, 0, 0.7071]) )
        # self.cabinet_scene.build(build_config_idxs=[1])
        
        # If you previously built the full robocasa scene, skip it and use this:
        # programmatic open cabinet only:
        # size is width, depth, height (meters)
        cab_size = [0.6, 0.4, 0.9]  # adjust to taste
        open_cab = OpenCabinet(
            scene=self.scene,
            name="open_cabinet",
            size=cab_size,
            num_shelves=3,
            thickness=0.03,
            texture=None,
            pos=[-0.2, 0.5, 0.2],  # center pos in world coordinates (x,y,z)
            rng=np.random.default_rng(),  # or pass your env rng
        )
        # Build into the scene (for single env index, pass [0]; for batched envs use proper indices):
        open_cab.quat = sapien.Pose(q=[0.7071,0,0,-0.7071]).q  # default orientation
        open_cab.pos = np.array([0.25, -0.12, 0.456])
        # choose scene indices to build into; if you have a batch, build into all relevant indices
        built = open_cab.build(scene_idxs=[0])
        # If environment uses multiple envs, repeat build for each environment index you care about.
        # Optionally keep a handle:
        self.open_cabinet = built
        self.left = actors.build_box(
            self.scene,
            half_sizes=[0.38/2, 0.01, 0.272],
            color=np.array([141, 117, 105, 255]) / 255,
            name="left",
            body_type="static",
            initial_pose=sapien.Pose(p=[0.252629, 0.195302, 0.309642]),
        )
        self.right = actors.build_box(
            self.scene,
            half_sizes=[0.38/2, 0.01, 0.272],
            color=np.array([141, 117, 105, 255]) / 255,
            name="right",
            body_type="static",
            initial_pose=sapien.Pose(p=[0.252629, -0.436221, 0.309642]),
        )
        self.back = actors.build_box(
            self.scene,
            half_sizes=[0.58/2, 0.01, 0.272],
            color=np.array([141, 117, 105, 255]) / 255,
            name="back",
            body_type="static",
            initial_pose=sapien.Pose(p=[0.252629, -0.436221, 0.309642]),
        )
        self.soda = self.load_glb_as_actor(self.scene, 
                                             "/home/prajwal-vijay/Downloads/ManiSkill-main/mani_skill/assets/place_soda_in_cabinet/opened_soda_can.glb",
                                            sapien.Pose(p=[0.055, -0.158, 0.1], q=[0.854,0.471,0.212,0.068]),
                                            name="soda_can",
                                            scale=[0.04,0.04,0.04],
                                            type="dynamic")
        
    @staticmethod
    def load_glb_as_actor(scene, glb_file_path, pose, name, scale, type="static"):
        """Load GLB file as a static actor in the scene"""
        builder = scene.create_actor_builder()
        builder.add_visual_from_file(glb_file_path, scale=scale)
        builder.add_multiple_convex_collisions_from_file(glb_file_path, decomposition="coacd", scale=scale)
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
            xyz[:, 2] = 0.405
            # xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[0.08, -0.26],[0.162, 0.12]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            soda_xy = sampler.sample(radius, 100)
            # # cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = soda_xy
            # qs = randomization.random_quaternions(
            #     b,
            #     lock_x=True,
            #     lock_y=True,
            #     lock_z=True,
            # )
            # [0.854,0.471,0.212,0.068] - q for sleeping book
            # [0.748, 0.279, -0.464, 0.384] - q for other side facing book
            self.soda.set_pose(Pose.create_from_pq(p=xyz.clone(), q=torch.tensor([0.0, 0, 0.7071, 0.7071]).repeat(b,1)))
            self.left.set_pose(Pose.create_from_pq(p=torch.tensor([0.254005, 0.177265, 0.309642]), q=torch.tensor([1,0,0,0]).repeat(b,1)))
            self.right.set_pose(Pose.create_from_pq(p=torch.tensor([0.254005, -0.422210, 0.309642]), q=torch.tensor([1,0,0,0]).repeat(b,1)))
            self.back.set_pose(Pose.create_from_pq(p=torch.tensor([0.44, -0.120, 0.309642]), q=torch.tensor([0.7071,0,0,-0.7071]).repeat(b,1)))
            # xyz[:, :2] = cubeB_xy
            # qs = randomization.random_quaternions(
            #     b,
            #     lock_x=True,
            #     lock_y=True,
            #     lock_z=False,
            # )
            # self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            # return
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
        is_soda_static = self.soda.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        print(self.soda.pose.p)
        is_soda_on_table = self.soda.pose.p[0][2] < 0.1
        success = is_soda_static * (is_soda_on_table)
        return {
            # "is_book_grasped": is_book_grasped,
            "is_soda_on_table": is_soda_on_table,
            "is_soda_static": is_soda_static,
            "success": success.bool()
        }
        
    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cabinet_pose=self.open_cabinet.pose.raw_pose,
                soda_pose=self.soda.pose.raw_pose,
                tcp_to_cabinet_pos=self.open_cabinet.pose.p - self.agent.tcp.pose.p,
                tcp_to_soda_pos=self.soda.pose.p - self.agent.tcp.pose.p,
                # book_to_shelf_pos=self.shelf.pose.p - self.book_A.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # rotation reward as cosine similarity between peg direction vectors
        # peg center of mass to end of peg, (1,0,0), rotated by peg pose rotation
        # dot product with its goal orientation: (0,0,1) or (0,0,-1)
        qmats = rotation_conversions.quaternion_to_matrix(self.soda.pose.q)
        vec = torch.tensor([-1.0, 0, 0], device=self.device)
        goal_vec = torch.tensor([0, 0, 1.0], device=self.device)
        rot_vec = (qmats @ vec).view(-1, 3)
        # abs since (0,0,-1) is also valid, values in [0,1]
        rot_rew = (rot_vec @ goal_vec).view(-1).abs()
        reward = rot_rew

        # position reward using common maniskill distance reward pattern
        # giving reward in [0,1] for moving center of mass toward half length above table
        z_dist = torch.abs(self.soda.pose.p[:, 2] - 0.16)
        reward += 1 - torch.tanh(5 * z_dist)

        # small reward to motivate initial reaching
        # initially, we want to reach and grip peg
        to_grip_vec = self.soda.pose.p - self.agent.tcp.pose.p
        to_grip_dist = torch.linalg.norm(to_grip_vec, axis=1)
        reaching_rew = 1 - torch.tanh(5 * to_grip_dist)
        # reaching reward granted if gripping block
        reaching_rew[self.agent.is_grasping(self.soda)] = 1
        # weight reaching reward less
        reaching_rew = reaching_rew / 5
        reward += reaching_rew

        reward[info["success"]] = 3

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8

    def _check_robocasa_dataset(self):
        """
        Check if RoboCasa dataset is available.
        If not, provide helpful error message with download instructions.
        """
        import pathlib
        
        # Check for a key RoboCasa fixture file
        robocasa_data_path = pathlib.Path.home() / ".maniskill" / "data" / "scene_datasets" / "robocasa_dataset"
        cabinet_fixture_path = robocasa_data_path / "assets" / "fixtures" / "cabinets" / "cabinet_open.xml"
        
        if not cabinet_fixture_path.exists():
            error_msg = f"""
================================================================================
ERROR: RoboCasa dataset not found!
================================================================================

The PickSodaFromCabinet-v1 environment requires the RoboCasa dataset.

Expected location: {robocasa_data_path}
Missing file: {cabinet_fixture_path}

To download the dataset, run the following commands:

    # Using ManiSkill's dataset download tool
    python -m mani_skill.utils.download_asset robocasa_dataset
    
OR manually:

    # Navigate to your ManiSkill data directory
    mkdir -p ~/.maniskill/data/scene_datasets
    cd ~/.maniskill/data/scene_datasets
    
    # Download the RoboCasa dataset (adjust URL as needed)
    # Check the official ManiSkill/RoboCasa documentation for the correct link
    
After downloading, verify the path exists:
    ls -la ~/.maniskill/data/scene_datasets/robocasa_dataset/assets/fixtures/cabinets/

For more information, see:
    - https://github.com/haosulab/ManiSkill
    - https://github.com/haosulab/RoboCasa

================================================================================
"""
            raise FileNotFoundError(error_msg)
