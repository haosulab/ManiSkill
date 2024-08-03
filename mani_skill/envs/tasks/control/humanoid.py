"""Adapted from https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/humanoid.py"""

from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.humanoid import Humanoid
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization, rewards
from mani_skill.sensors.camera import CameraConfig, update_camera_cfgs_from_dict
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, SceneConfig, SimConfig

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.4

# Horizontal speeds above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 10

###################################################################
class HumanoidEnvBase(BaseEnv):
    agent: Union[Humanoid]

    def __init__(self, *args, robot_uids="humanoid", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            scene_cfg=SceneConfig(
                solver_position_iterations=4, solver_velocity_iterations=1
            ),
            sim_freq=200,
            control_freq=40,
        )

    @property
    def _default_sensor_configs(self):
        return [
            # replicated from xml file
            CameraConfig(
                uid="cam0",
                pose=sapien_utils.look_at(eye=[0, -2.8, 0.8], target=[0, 0, 0]),
                width=128,
                height=128,
                fov=np.pi / 4,
                near=0.01,
                far=100,
                mount=self.camera_mount,
            ),
        ]

    def _before_control_step(self):
        self.camera_mount.set_pose(
            Pose.create_from_pq(p=self.agent.robot.links_map["torso"].pose.p)
        )
        if sapien.physx.is_gpu_enabled():
            # we update just actor pose here, no need to call apply_all/fetch_all
            self.scene.px.gpu_apply_rigid_dynamic_data()
            self.scene.px.gpu_fetch_rigid_dynamic_data()

    @property
    def _default_human_render_camera_configs(self):
        return [
            # replicated from xml file
            CameraConfig(
                uid="not_render_cam",
                pose=sapien_utils.look_at(eye=[0, -3, 1], target=[0, 0, 0]),
                width=512,
                height=512,
                fov=60 * np.pi / 180,
                near=0.01,
                far=100,
                mount=self.camera_mount,
            ),
        ]

    @property
    def head_height(self):
        """Returns the height of the head."""
        return self.agent.robot.links_map["head"].pose.p[:, -1]

    def torso_upright(self, info):
        # print("rot_mat")
        # print(self.agent.robot.links_map["torso"].pose.to_transformation_matrix())
        # print("rot_mat_zz", self.agent.robot.links_map["torso"].pose.to_transformation_matrix()[:,2,2])
        # print("rot_mat.shape", self.agent.robot.links_map["torso"].pose.to_transformation_matrix().shape)
        return info["torso_xmat"][:, 2, 2]

    # not sure this is correct - are our pose rotations the same as mujocos?
    # test out the transpose (taking column instead of row)
    # right now, it should represnt the z axis of global in the local frame - hm, no, this should be correct
    def torso_vertical_orientation(self, info):
        return info["torso_xmat"][:, 2, :3].view(-1, 3)

    def extremities(self, info):
        torso_frame = info["torso_xmat"][:, :3, :3].view(-1, 3, 3)
        torso_pos = self.agent.robot.links_map["torso"].pose.p
        positions = []
        for side in ("left_", "right_"):
            for limb in ("hand", "foot"):
                torso_to_limb = (
                    self.agent.robot.links_map[side + limb].pose.p - torso_pos
                ).view(-1, 1, 3)
                positions.append(
                    (torso_to_limb @ torso_frame).view(-1, 3)
                )  # reverse order mult == extrems in torso frame
        return torch.stack(positions, dim=1).view(-1, 12)  # (b, 4, 3) -> (b,12)

    @property
    def center_of_mass_velocity(self):
        # """Returns the center of mass velocity of robot"""
        vels = torch.stack(
            [link.get_linear_velocity() for link in self.active_links], dim=0
        )  # (num_links, b, 3)
        vels *= self.robot_links_mass.view(-1, 1, 1)
        com_vel = vels.sum(dim=0) / self.robot_mass  # (b, 3)

        return com_vel

    def evaluate(self) -> Dict:
        return dict(
            torso_xmat=self.agent.robot.links_map[
                "torso"
            ].pose.to_transformation_matrix(),
            cmass_linvel=self.center_of_mass_velocity,
        )

    def _load_scene(self, options: dict):
        loader = self.scene.create_mjcf_loader()
        articulation_builders, actor_builders, sensor_configs = loader.parse(
            self.agent.mjcf_path
        )
        for a in actor_builders:
            a.build(a.name)

        self.ground = build_ground(self.scene)

        # allow tracking of humanoid
        self.camera_mount = self.scene.create_actor_builder().build_kinematic(
            "camera_mount"
        )

        # cache for com velocity calc - doing so gives + 10 fps boost for 1024 envs
        self.active_links = [
            link for link in self.agent.robot.get_links() if "dummy" not in link.name
        ]
        self.robot_links_mass = torch.stack(
            [link.mass[0] for link in self.active_links]
        ).to(self.device)
        self.robot_mass = torch.sum(self.robot_links_mass).item()

    def standing_rew(self):
        return rewards.tolerance(
            self.head_height,
            lower=_STAND_HEIGHT,
            upper=float("inf"),
            margin=_STAND_HEIGHT / 4,
        ).view(-1)

    def upright_rew(self, info: Dict):
        return rewards.tolerance(
            self.torso_upright(info),
            lower=0.9,
            upper=float("inf"),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        ).view(-1)

    def control_rew(self, action: Array):
        return (
            rewards.tolerance(action, margin=1, value_at_margin=0, sigmoid="quadratic")
            .mean(dim=-1)
            .view(-1)
        )  # (b, a) -> (b)

    def dont_move_rew(self, info):
        return (
            rewards.tolerance(info["cmass_linvel"][:, :2], margin=2)
            .mean(dim=-1)
            .view(-1)
        )  # (b,3) -> (b)

    def move_rew(self, info, move_speed=10):
        com_vel = torch.linalg.norm(info["cmass_linvel"][:, :2], dim=-1)
        return (
            rewards.tolerance(
                com_vel,
                lower=move_speed,
                upper=np.inf,
                margin=move_speed,
                value_at_margin=0,
                sigmoid="linear",
            )
            .mean(dim=-1)
            .view(-1)
        )  # (b,3) -> (b)

    # def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
    #     standing = rewards.tolerance(
    #         self.head_height,
    #         lower=_STAND_HEIGHT,
    #         upper=float("inf"),
    #         margin=_STAND_HEIGHT / 4,
    #     )
    #     upright = rewards.tolerance(
    #         self.torso_upright(info),
    #         lower=0.9,
    #         upper=float("inf"),
    #         sigmoid="linear",
    #         margin=1.9,
    #         value_at_margin=0,
    #     )
    #     stand_reward = standing.view(-1) * upright.view(-1)
    #     small_control = rewards.tolerance(
    #         action, margin=1, value_at_margin=0, sigmoid="quadratic"
    #     ).mean(
    #         dim=-1
    #     )  # (b, a) -> (b)
    #     small_control = (4 + small_control) / 5
    #     horizontal_velocity = self.center_of_mass_velocity[:, :2]
    #     dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean(
    #         dim=-1
    #     )  # (b,3) -> (b)

    #     return small_control.view(-1) * stand_reward * dont_move.view(-1)

    # def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
    #     max_reward = 1.0
    #     return self.compute_dense_reward(obs, action, info) / max_reward


###
class HumanoidEnvStandard(HumanoidEnvBase):
    agent: Union[Humanoid]

    def __init__(self, *args, robot_uids="humanoid", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _get_obs_state_dict(self, info: Dict):
        # our qpos model doesn't include the free joint, meaning qpos and qvel are 21 dims, not 27
        # global dpos/dt and root (torso) dquaterion/dt lost
        # we replace with linear root linvel and root angularvel (equivalent info)
        return dict(
            agent=self._get_obs_agent(),  # (b, 21*2) root joint not included in our qpos
            root_vel=self.agent.robot.links_map[
                "dummy_root_0"
            ].get_linear_velocity(),  # free joint info, (b, 3)
            root_quat_vel=self.agent.robot.links_map[
                "dummy_root_0"
            ].get_angular_velocity(),  # free joint info, (b, 3)
            head_height=self.head_height,  # (b,1)
            com_velocity=info["cmass_linvel"],  # (b, 3)
            extremities=self.extremities(info),
            link_linvels=torch.stack(
                [link.get_linear_velocity() for link in self.active_links], dim=1
            ).view(-1, 16 * 3),
            link_angvels=torch.stack(
                [link.get_angular_velocity() for link in self.active_links], dim=1
            ).view(-1, 16 * 3),
            qfrc=self.agent.robot.get_qf(),
        )

    # our standard humanoid env resets if torso is below a certain point
    def _load_scene(self, options: Dict):
        super()._load_scene(options)
        self.ground.set_collision_group_bit(group=2, bit_idx=30, bit=1)

        for link in self.active_links:
            if not "foot" in link.name:
                link.set_collision_group_bit(group=2, bit_idx=30, bit=1)

    def evaluate(self) -> Dict:
        info = super().evaluate()
        torso_z = self.agent.robot.links_map["torso"].pose.p[:, -1]
        failure = torch.logical_or(torso_z < 1.0, torso_z > 2.0)
        info.update(fail=failure)
        return info

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        # self.camera_mount.set_pose(Pose.create_from_pq(p=self.agent.robot.links_map["torso"].pose.p))
        with torch.device(self.device):
            b = len(env_idx)
            # set agent root pose - torso now centered at dummy root at (0,0,0)
            pose = Pose.create_from_pq(
                p=torch.tensor([0, 0, 1.3]).unsqueeze(0).repeat(b, 1)
            )
            self.agent.robot.set_root_pose(pose)
            # set randomized qpos
            noise_scale = 1e-2
            qpos_noise = (
                torch.rand(b, self.agent.robot.dof[0]) * (2 * noise_scale)
            ) - noise_scale
            qvel_noise = (
                torch.rand(b, self.agent.robot.dof[0]) * (2 * noise_scale)
            ) - noise_scale
            self.agent.robot.set_qpos(qpos_noise)
            self.agent.robot.set_qvel(qvel_noise)


@register_env("MS-HumanoidStand-v1", max_episode_steps=100)
class HumanoidStand(HumanoidEnvStandard):
    agent: Union[Humanoid]

    def __init__(self, *args, robot_uids="humanoid", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        small_control = (4 + self.control_rew(action)) / 5
        return (
            small_control
            * self.standing_rew()
            * self.upright_rew(info)
            * self.dont_move_rew(info)
        )

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info)


class HumanoidEnvHard(HumanoidEnvBase):
    agent: Union[Humanoid]

    def __init__(self, *args, robot_uids="humanoid", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _get_obs_state_dict(self, info: Dict):
        # our qpos model doesn't include the free joint, meaning qpos and qvel are 21 dims, not 27
        # global dpos/dt and root (torso) dquaterion/dt lost
        # we replace with linear root linvel and root angularvel (equivalent info)
        return dict(
            agent=self._get_obs_agent(),  # (b, 21*2) root joint not included in our qpos
            head_height=self.head_height,  # (b,1)
            extremities=self.extremities(info),  # (b, 12)
            torso_vertical=self.torso_vertical_orientation(info),  # (b, 3)
            com_velocity=info["cmass_linvel"],  # (b, 3)
            root_vel=self.agent.robot.links_map[
                "dummy_root_0"
            ].get_linear_velocity(),  # free joint info, (b, 3)
            root_quat_vel=self.agent.robot.links_map[
                "dummy_root_0"
            ].get_angular_velocity(),  # free joint info, (b, 3)
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        self.camera_mount.set_pose(
            Pose.create_from_pq(p=self.agent.robot.links_map["torso"].pose.p)
        )
        with torch.device(self.device):
            b = len(env_idx)
            # set agent root pose - torso now centered at dummy root at (0,0,0)
            pose = Pose.create_from_pq(
                p=[0, 0, 1.5], q=randomization.random_quaternions(b)
            )
            self.agent.robot.set_root_pose(pose)
            # set randomized qpos
            random_qpos = torch.rand(b, self.agent.robot.dof[0])
            q_lims = self.agent.robot.get_qlimits()
            q_ranges = q_lims[..., 1] - q_lims[..., 0]
            random_qpos *= q_ranges
            random_qpos += q_lims[..., 0]
            self.agent.reset(random_qpos)


###
@register_env("MS-HumanoidStandHard-v1", max_episode_steps=100)
class HumanoidStandHard(HumanoidEnvHard):
    agent: Union[Humanoid]

    def __init__(self, *args, robot_uids="humanoid", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        small_control = (4 + self.control_rew(action)) / 5
        return (
            small_control
            * self.standing_rew()
            * self.upright_rew(info)
            * self.dont_move_rew(info)
        )

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info)


@register_env("MS-HumanoidWalkHard-v1", max_episode_steps=100)
class HumanoidWalkHard(HumanoidEnvHard):
    agent: Union[Humanoid]

    def __init__(self, *args, robot_uids="humanoid", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        small_control = (4 + self.control_rew(action)) / 5
        return (
            small_control
            * self.standing_rew()
            * self.upright_rew(info)
            * self.move_rew(info, _WALK_SPEED)
        )

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info)


@register_env("MS-HumanoidRunHard-v1", max_episode_steps=100)
class HumanoidRunHard(HumanoidEnvHard):
    agent: Union[Humanoid]

    def __init__(self, *args, robot_uids="humanoid", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        small_control = (4 + self.control_rew(action)) / 5
        return (
            small_control
            * self.standing_rew()
            * self.upright_rew(info)
            * self.move_rew(info, _RUN_SPEED)
        )

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info)
