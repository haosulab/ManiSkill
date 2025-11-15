import os
from typing import Any, Optional, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization, rewards
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, SceneConfig, SimConfig

_STAND_HEIGHT = 0.55
_WALK_SPEED = 0.5
_RUN_SPEED = 4

MJCF_FILE = f"{os.path.join(os.path.dirname(__file__), 'assets/ant.xml')}"


class AntRobot(BaseAgent):
    uid = "ant"
    mjcf_path = MJCF_FILE
    fix_root_link = False

    keyframes = dict(
        stand=Keyframe(
            qpos=np.array([0, 0, 0, 0, 1, -1, -1, 1]),
            pose=sapien.Pose(p=[0, 0, -0.175], q=euler2quat(0, 0, np.pi / 2)),
        )
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        body = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.get_active_joints()],
            lower=-1,
            upper=1,
            damping=1e2,
            stiffness=1e3,
            use_delta=True,
        )
        return deepcopy_dict(
            dict(
                pd_joint_delta_pos=dict(
                    body=body,
                    balance_passive_force=False,
                ),
            )
        )

    def _load_articulation(
        self, initial_pose: Optional[Union[sapien.Pose, Pose]] = None
    ):
        """
        Load the robot articulation
        """
        loader = self.scene.create_mjcf_loader()
        asset_path = str(self.mjcf_path)

        loader.name = self.uid

        builder = loader.parse(asset_path)["articulation_builders"][0]
        builder.initial_pose = initial_pose
        self.robot = builder.build()
        assert self.robot is not None, f"Fail to load URDF/MJCF from {asset_path}"
        self.robot_link_ids = [link.name for link in self.robot.get_links()]


class AntEnv(BaseEnv):
    agent: Union[AntRobot]
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "none")

    def __init__(self, *args, robot_uids=AntRobot, move_speed=0, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self.move_speed = move_speed

    @property
    def _default_sim_config(self):
        return SimConfig(
            scene_config=SceneConfig(
                solver_position_iterations=4, solver_velocity_iterations=1
            ),
            spacing=20,
            sim_freq=200,
            control_freq=40,
        )

    @property
    def _default_sensor_configs(self):
        return [
            CameraConfig(
                uid="side_cam",
                pose=sapien_utils.look_at(eye=[0.5, -2, 1], target=[0, 0, 0]),
                width=128,
                height=128,
                fov=60 * np.pi / 180,
                near=0.01,
                far=100,
                mount=self.camera_mount,
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        return [
            CameraConfig(
                uid="training_side_vis",
                pose=sapien_utils.look_at(eye=[0.5, -2, 1], target=[0, 0, 0]),
                width=512,
                height=512,
                fov=60 * np.pi / 180,
                near=0.01,
                far=100,
                mount=self.camera_mount,
            ),
        ]

    def _load_scene(self, options: dict):
        loader = self.scene.create_mjcf_loader()
        actor_builders = loader.parse(self.agent.mjcf_path)["actor_builders"]
        for a in actor_builders:
            a.build(a.name)

        self.ground = build_ground(self.scene, floor_width=500)

        # allow tracking of ant
        self.camera_mount = self.scene.create_actor_builder().build_kinematic(
            "camera_mount"
        )

        # cache for com velocity calc
        self.active_links = [
            link for link in self.agent.robot.get_links() if "dummy" not in link.name
        ]
        self.robot_links_mass = torch.stack(
            [link.mass[0] for link in self.active_links]
        ).to(self.device)
        self.robot_mass = torch.sum(self.robot_links_mass).item()

        self.force_sensor_links = [
            link.name for link in self.active_links if "foot" in link.name
        ]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            # set agent root pose - torso now centered at dummy root at (0,0,0)
            self.agent.robot.set_root_pose(self.agent.keyframes["stand"].pose)

            # set randomized qpos
            base_qpos = (
                torch.tensor(self.agent.keyframes["stand"].qpos, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(b, 1)
            )
            noise_scale = 1e-2
            qpos_noise = (
                torch.rand(b, self.agent.robot.dof[0]) * (2 * noise_scale)
            ) - noise_scale
            qvel_noise = (
                torch.rand(b, self.agent.robot.dof[0]) * (2 * noise_scale)
            ) - noise_scale
            self.agent.robot.set_qpos(base_qpos + qpos_noise)
            self.agent.robot.set_qvel(qvel_noise)

            # set the camera to begin tracking the agent
            self.camera_mount.set_pose(
                Pose.create_from_pq(p=self.agent.robot.links_map["torso"].pose.p)
            )

    # reset the camera mount every timstep
    # necessary because we want camera to follow torso pos only, not orientation
    def _after_control_step(self):
        self.camera_mount.set_pose(
            Pose.create_from_pq(p=self.agent.robot.links_map["torso"].pose.p)
        )
        # gpu requires that we manually apply this update
        if self.gpu_sim_enabled:
            # we update just actor pose here, no need to call apply_all/fetch_all
            self.scene.px.gpu_apply_rigid_dynamic_data()
            self.scene.px.gpu_fetch_rigid_dynamic_data()

    @property
    def get_vels(self):
        """Returns linvel and angvel of each link and cmass linvel of robot"""
        angvels = torch.stack(
            [link.get_angular_velocity() for link in self.active_links], dim=1
        )  # (num_links, b, 3)
        linvels = torch.stack(
            [link.get_linear_velocity() for link in self.active_links], dim=1
        )  # (num_links, b, 3)

        batch_angvels = angvels.view(-1, len(self.active_links) * 3)
        batch_linvels = linvels.view(-1, len(self.active_links) * 3)

        com_vel = linvels * self.robot_links_mass.view(1, -1, 1)
        com_vel = com_vel.sum(dim=1) / self.robot_mass  # (b, 3)

        return batch_angvels, batch_linvels, com_vel

    @property
    def torso_height(self):
        return self.agent.robot.links_map["torso"].pose.raw_pose[:, -1]

    @property
    def foot_contact_forces(self):
        """Returns log1p of force on foot links"""
        force_vecs = torch.stack(
            [
                self.agent.robot.get_net_contact_forces([link])
                for link in self.force_sensor_links
            ],
            dim=1,
        )
        force_mag = torch.linalg.norm(force_vecs, dim=-1).view(
            -1, len(self.force_sensor_links)
        )  # (b, len(self.force_sensor_links))
        return torch.log1p(force_mag)

    @property
    def link_orientations(self):
        return torch.stack([link.pose.q for link in self.active_links], dim=-1).view(
            -1, len(self.active_links) * 4
        )

    # cache re-used computation
    def evaluate(self) -> dict:
        link_angvels, link_linvels, cmass_linvel = self.get_vels
        return dict(
            link_angvels=link_angvels,
            link_linvels=link_linvels,
            cmass_linvel=cmass_linvel,
        )

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        if self.obs_mode_struct.use_state:
            obs.update(
                cmass=info["cmass_linvel"],
                link_angvels=info["link_angvels"],
                link_linvels=info["link_linvels"],
                height=self.torso_height.view(-1, 1),
                link_orientations=self.link_orientations,
                foot_contact_forces=self.foot_contact_forces,
            )
        return obs

    def move_x_rew(self, info, move_speed=_WALK_SPEED):
        com_vel_x = info["cmass_linvel"][:, 0]
        return rewards.tolerance(
            com_vel_x,
            lower=move_speed,
            upper=np.inf,
            margin=move_speed,
            value_at_margin=0,
            sigmoid="linear",
        ).view(-1)

    def standing_rew(self):
        return rewards.tolerance(
            self.torso_height,
            lower=_STAND_HEIGHT,
            upper=float("inf"),
            margin=_STAND_HEIGHT / 4,
        ).view(-1)

    def control_rew(self, action: Array):
        return (
            rewards.tolerance(action, margin=1, value_at_margin=0, sigmoid="quadratic")
            .mean(dim=-1)
            .view(-1)
        )

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        small_control = (4 + self.control_rew(action)) / 5
        return (
            small_control * self.move_x_rew(info, self.move_speed) * self.standing_rew()
        )

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs, action, info)


@register_env("MS-AntWalk-v1", max_episode_steps=1000)
class AntWalk(AntEnv):
    """
    **Task Description:**
    Ant moves in x direction at 0.5 m/s

    **Randomizations:**
    - Ant qpos and qvel have added noise from uniform distribution [-1e-2, 1e-2]

    **Success Conditions:**
    - No specific success conditions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids=AntRobot, move_speed=_WALK_SPEED, **kwargs)


@register_env("MS-AntRun-v1", max_episode_steps=1000)
class AntRun(AntEnv):
    """
    **Task Description:**
    Ant moves in x direction at 4 m/s

    **Randomizations:**
    - Ant qpos and qvel have added noise from uniform distribution [-1e-2, 1e-2]

    **Success Conditions:**
    - No specific success conditions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids=AntRobot, move_speed=_RUN_SPEED, **kwargs)
