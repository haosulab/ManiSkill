from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots.koch.koch import Koch
from mani_skill.envs.tasks.digital_twins.base_env import BaseDigitalTwinEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose


# there are many ways to parameterize an environment's domain randomization. This is a simple way to do it
# with dataclasses that can be created and modified by the user and passed into the environment constructor.
@dataclass
class KochGraspCubeDomainRandomizationConfig:
    cube_half_size_range: Tuple[float, float] = (0.01, 0.015)


@register_env("KochGraspCube-v1", max_episode_steps=50)
class KochGraspCubeEnv(BaseDigitalTwinEnv):
    """
    **Task Description:**
    A simple task where the objective is to grasp a cube with the Koch arm and bring it up to a target rest pose.

    **Randomizations:**
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the cube's z-axis rotation is randomized to a random angle
    - the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

    **Success Conditions:**
    - the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "koch-v1.1",
    ]
    SUPPORTED_OBS_MODES = ["state", "state_dict", "rgb+segmentation"]
    agent: Koch

    def __init__(
        self,
        *args,
        robot_uids="koch-v1.1",
        greenscreen_overlay_path="/home/stao/.maniskill/data/tasks/bridge_v2_real2sim_dataset/real_inpainting/bridge_real_eval_1.png",
        domain_randomization_config=KochGraspCubeDomainRandomizationConfig(),
        domain_randomization=False,
        **kwargs,
    ):
        self.domain_randomization = domain_randomization
        self.domain_randomization_config = domain_randomization_config

        # set the camera called "base_camera" to use the greenscreen overlay when rendering
        self.rgb_overlay_paths = dict(base_camera=greenscreen_overlay_path)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        # we just set a default camera pose here for now. For sim2real we will modify this during training accordingly.
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # this camera and angle is simply used for visualization purposes, not policy observations
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        # load the koch arm at this initial pose
        super()._load_agent(
            options, sapien.Pose(p=[-0.615, 0, 0], q=euler2quat(0, 0, np.pi / 2))
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=0.02 if self.domain_randomization else 0.0
        )
        self.table_scene.build()

        # randomize cube sizes, colors, and frictions # TODO
        if self.domain_randomization:
            half_sizes = self._batched_episode_rng.uniform(
                low=self.domain_randomization_config.cube_half_size_range[0],
                high=self.domain_randomization_config.cube_half_size_range[1],
            )
            colors = self._batched_episode_rng.uniform(
                low=[0, 0, 0], high=[1, 1, 1], size=(self.num_envs, 3)
            )
        else:
            half_sizes = (
                np.ones(self.num_envs)
                * (
                    self.domain_randomization_config.cube_half_size_range[1]
                    + self.domain_randomization_config.cube_half_size_range[0]
                )
                / 2
            )
            colors = np.zeros((self.num_envs, 3))
            colors[:, 0] = 1

        self.cube_half_sizes = common.to_tensor(half_sizes, device=self.device)
        colors = np.concatenate([colors, np.ones((self.num_envs, 1))], axis=-1)

        cubes = []
        for i in range(self.num_envs):
            cube = actors.build_cube(
                self.scene,
                half_size=half_sizes[i],
                color=colors[i],
                name=f"cube-{i}",
                initial_pose=sapien.Pose(p=[0, 0, half_sizes[i]]),
            )
            cubes.append(cube)
            self.remove_from_state_dict_registry(cube)
        self.cube = Actor.merge(cubes)
        self.add_to_state_dict_registry(self.cube)

        # we want to only keep the robot and the cube in the render, everything else is greenscreened.
        self.remove_object_from_greenscreen(self.agent.robot)
        self.remove_object_from_greenscreen(self.cube)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # we randomize the pose of the cube accordingly so that the policy can learn to pick up the cube from
        # many different orientations and positions.
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.cube_half_sizes
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            self.agent.robot.set_qpos(
                np.array([0.0, 2.2, 2.75, -0.25, -np.pi / 2, 1.0])
            )

    def _get_obs_agent(self):
        # we remove qvel as koch arm qvel is too noisy to learn from and not implemented.
        obs = dict(qpos=self.agent.robot.get_qpos())
        controller_state = self.agent.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def _get_obs_extra(self, info: Dict):
        # we ensure that the observation data is always retrievable in the real world, using only real world
        # available data (joint positions or the controllers target joint positions in this case).
        target_qpos = self.agent.controller._target_qpos.clone()
        is_grasped = (
            (self.agent.robot.qpos[..., -1] - target_qpos[..., -1]) >= 0.02
        ).float() * (target_qpos[..., -1] < 0.24)
        obs = dict(is_grasped=is_grasped)
        if self.obs_mode_struct.state:
            # state based policies can gain access to more information that helps learning
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_pose=self.agent.tcp.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def evaluate(self):
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_robot_static,
            # "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = obs["extra"]["is_grasped"]
        reward += is_grasped

        # obj_to_goal_dist = torch.linalg.norm(
        #     self.goal_site.pose.p - self.cube.pose.p, axis=1
        # )
        # place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        # reward += place_reward * is_grasped

        # qvel_without_gripper = self.agent.robot.get_qvel()
        # if self.robot_uids == "xarm6_robotiq":
        #     qvel_without_gripper = qvel_without_gripper[..., :-6]
        # elif self.robot_uids == "panda":
        #     qvel_without_gripper = qvel_without_gripper[..., :-2]
        # static_reward = 1 - torch.tanh(
        #     5 * torch.linalg.norm(qvel_without_gripper, axis=1)
        # )
        # reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
