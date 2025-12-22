from typing import Any, Dict, Union
import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.distraction_set import DistractionSet


@register_env("PlaceAppleOnPlate-v1", max_episode_steps=50)
class PlaceAppleOnPlateEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    APPLE_RADIUS = 0.04
    PLATE_RADIUS = 0.13
    PLATE_HEIGHT = 0.02

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):        
        distraction_set: Union[DistractionSet, dict] = kwargs.pop("distraction_set")
        self._distraction_set: DistractionSet = DistractionSet(**distraction_set) if isinstance(distraction_set, dict) else distraction_set
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.6, -0.7, 0.6], [0.0, 0.0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self._load_apple()
        self._load_plate()

    def _load_apple(self):
        try:
            apple_builder = actors.get_actor_builder(self.scene, id="ycb:013_apple")
            apple_builder.initial_pose = sapien.Pose(p=[0, 0, self.APPLE_RADIUS])
            self.apple = apple_builder.build(name="apple")
        except Exception as e:
            self.apple = actors.build_sphere(
                self.scene,
                radius=self.APPLE_RADIUS,
                color=[1, 0, 0, 1],
                name="apple",
                initial_pose=sapien.Pose(p=[0, 0, self.APPLE_RADIUS]),
            )

    def _load_plate(self):
        try:
            plate_builder = actors.get_actor_builder(self.scene, id="ycb:029_plate")
            plate_builder.initial_pose = sapien.Pose(p=[0.2, 0, self.PLATE_HEIGHT / 2])
            self.plate = plate_builder.build_static(name="plate")
        except Exception as e:
            self.plate = actors.build_cylinder(
                self.scene,
                radius=self.PLATE_RADIUS,
                half_length=self.PLATE_HEIGHT / 2,
                color=[0.9, 0.9, 0.9, 1],
                name="plate",
                initial_pose=sapien.Pose(p=[0.2, 0, self.PLATE_HEIGHT / 2]),
                body_type="static",
            )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )

            plate_radius_with_margin = self.PLATE_RADIUS + 0.02
            plate_xy = sampler.sample(plate_radius_with_margin, 100)
            
            plate_xyz = torch.zeros((b, 3), device=self.device)
            plate_xyz[:, :2] = plate_xy
            plate_xyz[:, 2] = self.PLATE_HEIGHT / 2
            self.plate.set_pose(Pose.create_from_pq(p=plate_xyz))

            apple_radius_with_margin = self.APPLE_RADIUS + 0.02
            apple_xy = sampler.sample(apple_radius_with_margin, 100, verbose=False)
            
            apple_xyz = torch.zeros((b, 3), device=self.device)
            apple_xyz[:, :2] = apple_xy
            apple_xyz[:, 2] = self.APPLE_RADIUS
            self.apple.set_pose(Pose.create_from_pq(p=apple_xyz))

    def evaluate(self):
        pos_apple = self.apple.pose.p
        pos_plate = self.plate.pose.p

        offset = pos_apple - pos_plate
        horizontal_dist = torch.sqrt(offset[..., 0]**2 + offset[..., 1]**2)
        is_apple_on_plate_xy = horizontal_dist <= (self.PLATE_RADIUS - self.APPLE_RADIUS / 2)

        expected_z = self.PLATE_HEIGHT + self.APPLE_RADIUS
        z_dist = torch.abs(pos_apple[..., 2] - expected_z)
        is_apple_on_plate_z = z_dist <= 0.02

        is_apple_on_plate = is_apple_on_plate_xy & is_apple_on_plate_z
        is_apple_static = self.apple.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_apple_grasped = self.agent.is_grasping(self.apple)
        success = is_apple_on_plate & is_apple_static & (~is_apple_grasped)

        return {
            "is_apple_grasped": is_apple_grasped,
            "is_apple_on_plate": is_apple_on_plate,
            "is_apple_static": is_apple_static,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                apple_pose=self.apple.pose.raw_pose,
                plate_pose=self.plate.pose.raw_pose,
                tcp_to_apple_pos=self.apple.pose.p - self.agent.tcp.pose.p,
                tcp_to_plate_pos=self.plate.pose.p - self.agent.tcp.pose.p,
                apple_to_plate_pos=self.plate.pose.p - self.apple.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        reward = torch.zeros(self.num_envs, device=self.device)

        tcp_pos = self.agent.tcp.pose.p
        apple_pos = self.apple.pose.p
        plate_pos = self.plate.pose.p

        tcp_to_apple_dist = torch.linalg.norm(tcp_pos - apple_pos, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_apple_dist)
        
        is_grasping = self.agent.is_grasping(self.apple)
        reaching_reward[is_grasping] = 1.0
        reward += reaching_reward

        grasp_reward = is_grasping.float() * 0.5
        reward += grasp_reward

        apple_to_plate_dist = torch.linalg.norm(
            apple_pos[:, :2] - plate_pos[:, :2], axis=1
        )
        placing_reward = 1 - torch.tanh(5 * apple_to_plate_dist)
        placing_reward = placing_reward * is_grasping.float()
        reward += placing_reward

        target_z = self.PLATE_HEIGHT + self.APPLE_RADIUS
        z_dist = torch.abs(apple_pos[:, 2] - target_z)
        height_reward = 1 - torch.tanh(5 * z_dist)
        height_reward = height_reward * (apple_to_plate_dist < self.PLATE_RADIUS).float()
        reward += height_reward * 0.5

        reward[info["success"]] = 5.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 5.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward