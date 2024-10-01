# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import torch
from collections.abc import Sequence

from omni.isaac.lab_assets.cartpole import CARTPOLE_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform


@configclass
class CartpoleRGBCameraBenchmarkEnvCfg(DirectRLEnvCfg):
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=2)

    # robot
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=128,
        height=128,
    )

    # change viewer settings
    viewer = ViewerCfg(eye=(20.0, 20.0, 20.0))

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=25.0, replicate_physics=True)

    # env
    decimation = 2
    episode_length_s = 5.0
    action_scale = 100.0  # [N]
    num_actions = 1
    num_channels = 3
    num_observations = num_channels * tiled_camera.height * tiled_camera.width
    num_states = 0

    # reset
    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.125, 0.125]  # the range in which the pole angle is sampled from on reset [rad]

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005

class CartpoleCameraBenchmarkEnv(DirectRLEnv):
    """Benchmark environment for CartPole task with a camera.

    Modification from original:
    - Remove reward / evaluation functions
    - Support RGB+Depth and multiple camera setups
    """

    cfg: CartpoleRGBCameraBenchmarkEnvCfg

    def __init__(
        self, cfg: CartpoleRGBCameraBenchmarkEnvCfg, render_mode: str | None = None, camera_width=128, camera_height=128, num_cameras=1, obs_mode="rgb", **kwargs
    ):
        # configure cameras
        data_types = []
        if "rgb" in obs_mode:
            data_types.append("rgb")
        if "depth" in obs_mode:
            data_types.append("depth")
        if "segmentation" in obs_mode:
            data_types.append("semantic_segmentation")
        self.data_types = data_types

        self.num_cameras = num_cameras
        self.tiled_camera_cfgs = []
        for i in range(num_cameras):
            tiled_camera_cfg = TiledCameraCfg(
                prim_path=f"/World/envs/env_.*/Camera_{i}",
                offset=TiledCameraCfg.OffsetCfg(pos=(-4.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
                data_types=data_types,
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 24.7)
                ),
                width=camera_width,
                height=camera_height,
            )
            self.tiled_camera_cfgs.append(tiled_camera_cfg)
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self._cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self._cartpole.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self._cartpole.data.joint_pos
        self.joint_vel = self._cartpole.data.joint_vel

        if len(self.cfg.tiled_camera.data_types) != 1:
            raise ValueError(
                "The Cartpole camera environment only supports one image type at a time but the following were"
                f" provided: {self.cfg.tiled_camera.data_types}"
            )

    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations
        self.num_states = self.cfg.num_states

        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["rgb"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_cameras, self.tiled_camera_cfgs[0].height, self.tiled_camera_cfgs[0].width, 3),
        )
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_actions,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # RL specifics
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)

    def _setup_scene(self):
        """Setup the scene with the cartpole and camera."""
        self._cartpole = Articulation(self.cfg.robot_cfg)
        # if self.has_rgb:
        self.tiled_cameras = [TiledCamera(cfg) for cfg in self.tiled_camera_cfgs]
        # if self.has_depth:
            # self.tiled_depth_cameras = [TiledCamera(cfg) for cfg in self.tiled_depth_camera_cfgs]

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(500, 500)))
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add articultion and sensors to scene
        self.scene.articulations["cartpole"] = self._cartpole
        for i in range(self.num_cameras):
            self.scene.sensors[f"tiled_camera_{i}"] = self.tiled_cameras[i]
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self._cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        # data_type = "rgb" if "rgb" in self.cfg.tiled_camera.data_types else "depth"
        # observations = {"policy": self._tiled_camera.data.output[data_type].clone()}
        observations = {"sensors": {}}
        for i in range(self.num_cameras):
            observations["sensors"][f"cam_{i}"] = {}
        for i, (cam, cfg) in enumerate(zip(self.tiled_cameras, self.tiled_camera_cfgs)):
            for data_type in self.data_types:
                observations["sensors"][f"cam_{i}"][data_type] = cam.data.output[data_type].clone()
        # if self.has_depth:
        #     for i, (cam, cfg) in enumerate(zip(self.tiled_depth_cameras, self.tiled_depth_camera_cfgs)):
        #         observations["sensors"][f"cam_{i}"]["depth"] = cam.data.output["depth"].clone()
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = torch.zeros((self.num_envs,), device=self.sim.device)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self._cartpole.data.joint_pos
        self.joint_vel = self._cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._cartpole._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self._cartpole.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self._cartpole.data.default_joint_vel[env_ids]

        default_root_state = self._cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self._cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
