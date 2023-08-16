import numpy as np
import sapien.core as sapien
import trimesh
from sapien.core import Pose
from scipy.spatial import distance as sdist
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import quat2mat

from mani_skill2.agents.robots.mobile_panda import MobilePandaDualArm
from mani_skill2.utils.common import np_random
from mani_skill2.utils.geometry import (
    angle_between_vec,
    get_axis_aligned_bbox_for_articulation,
    get_local_axis_aligned_bbox_for_link,
    transform_points,
)
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import get_entity_by_name, vectorize_pose
from mani_skill2.utils.trimesh_utils import get_actor_visual_mesh

from .base_env import MS1BaseEnv


@register_env("MoveBucket-v1", max_episode_steps=200)
class MoveBucketEnv(MS1BaseEnv):
    DEFAULT_MODEL_JSON = (
        "{PACKAGE_ASSET_DIR}/partnet_mobility/meta/info_bucket_train.json"
    )
    ASSET_UID = "partnet_mobility_bucket"

    def _register_render_cameras(self):
        cam_cfg = super()._register_render_cameras()
        cam_cfg.p = [0, 0, 4]
        cam_cfg.q = [0.70710678, 0.0, 0.70710678, 0.0]
        return cam_cfg

    # -------------------------------------------------------------------------- #
    # Reconfigure
    # -------------------------------------------------------------------------- #
    def _load_articulations(self):
        urdf_config = dict(
            material=dict(static_friction=0.5, dynamic_friction=0.5, restitution=0),
            density=1000,
        )
        self.bucket = self._load_partnet_mobility(
            fix_root_link=False,
            scale=self.model_info.get("scale", 1),
            urdf_config=urdf_config,
        )
        self.bucket.set_name(self.model_id)

        # Restrict the handle's range of motion
        lim = self.bucket.get_active_joints()[0].get_limits()
        v = (lim[0, 1] - lim[0, 0]) * 0.1
        lim[0, 0] += v
        lim[0, 1] -= v
        self.bucket.get_active_joints()[0].set_limits(lim)

        if self._reward_mode in ["dense", "normalized_dense"]:
            self._set_bucket_links_mesh()

    def _set_bucket_links_mesh(self):
        self.links_info = {}
        for link in self.bucket.get_links():
            mesh = get_actor_visual_mesh(link)
            if mesh is None:
                continue
            self.links_info[link.name] = [link, mesh]

    def _load_actors(self):
        super()._load_actors()

        # place a target platform on ground
        box_half_size = [0.3, 0.3, 0.1]
        builder = self._scene.create_actor_builder()

        white_diffuse = self._renderer.create_material()
        white_diffuse.base_color = [0.8, 0.8, 0.8, 1]
        white_diffuse.metallic = 0.0
        white_diffuse.roughness = 0.9
        white_diffuse.specular = 0
        builder.add_box_visual(half_size=box_half_size, material=white_diffuse)

        obj_material = self._scene.create_physical_material(0.5, 0.5, 0)
        builder.add_box_collision(
            half_size=box_half_size, material=obj_material, density=1000
        )
        self.target_platform = builder.build_static(name="target_platform")

        # balls
        R = 0.05
        self.balls_radius = R
        builder = self._scene.create_actor_builder()
        builder.add_sphere_collision(radius=R, density=1000)
        builder.add_sphere_visual(radius=R, color=[0, 1, 1])
        self.balls = []
        self.GX = self.GY = self.GZ = 1
        for i in range(self.GX * self.GY * self.GZ):
            actor = builder.build(name="ball_{:d}".format(i + 1))
            self.balls.append(actor)

    def _configure_agent(self):
        self._agent_cfg = MobilePandaDualArm.get_default_config()

    def _load_agent(self):
        self.agent = MobilePandaDualArm(
            self._scene, self._control_freq, self._control_mode, config=self._agent_cfg
        )

        links = self.agent.robot.get_links()
        self.left_tcp: sapien.Link = get_entity_by_name(links, "left_panda_hand_tcp")
        self.right_tcp: sapien.Link = get_entity_by_name(links, "right_panda_hand_tcp")

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    def _initialize_task(self):
        self.root_link = self.bucket.get_links()[0]

        self._set_target()
        self._initialize_bucket()
        self._initialize_robot()
        self._initialize_balls()
        if self._reward_mode in ["dense", "normalized_dense"]:
            self._set_bucket_links_pcd()

        for _ in range(25):
            self._scene.step()

    def _set_target(self):
        self.target_xy = np.zeros(2)
        target_orientation = 0
        target_q = euler2quat(target_orientation, 0, 0, "szyx")
        self.target_p = np.zeros(3)
        self.target_p[:2] = self.target_xy
        self.target_platform.set_pose(Pose(p=self.target_p, q=target_q))

    def _initialize_bucket(self):
        pose = Pose(p=[0, 0, 2], q=[1, 0, 0, 0])
        self.bucket.set_pose(pose)
        bb = np.array(
            get_axis_aligned_bbox_for_articulation(self.bucket)
        )  # bb in world

        # Sample a position around the target
        center = self.target_xy
        dist = self._episode_rng.uniform(low=0.8, high=1.2)
        theta = self._episode_rng.uniform(low=-np.pi, high=np.pi)
        self.init_bucket_to_target_theta = theta
        delta = np.array([np.cos(theta), np.sin(theta)]) * dist
        pos_xy = center + delta

        self.bucket_center_offset = (bb[1, 2] - bb[0, 2]) / 5
        self.bucket_body_link = self.bucket.get_active_joints()[0].get_parent_link()
        self.bb_local = np.array(
            get_local_axis_aligned_bbox_for_link(self.bucket_body_link)
        )
        self.center_local = (self.bb_local[0] + self.bb_local[1]) / 2

        pose.set_p([pos_xy[0], pos_xy[1], pose.p[2] - bb[0, 2]])

        # Sample an orientation for the bucket
        ax = ay = 0
        az = self._episode_rng.uniform(low=-np.pi, high=np.pi)
        q = euler2quat(ax, ay, az, "sxyz")
        pose.set_q(q)

        # Finalize bucket pose
        self.bucket.set_pose(pose)
        self.init_bucket_height = (
            self.bucket_body_link.get_pose()
            .transform(self.bucket_body_link.get_cmass_local_pose())
            .p[2]
        )

        # Finalize the bucket joint state
        self.bucket.set_qpos(self.bucket.get_qlimits()[:, 0])
        self.bucket.set_qvel(np.zeros(self.bucket.dof))

    def _initialize_robot(self):
        # Base position
        center = self.bucket.get_pose().p
        dist = self._episode_rng.uniform(low=0.6, high=0.8)
        theta = self._episode_rng.uniform(low=-0.4 * np.pi, high=0.4 * np.pi)
        theta += self.init_bucket_to_target_theta
        delta = np.array([np.cos(theta), np.sin(theta)]) * dist
        base_pos = center[:2] + delta

        # Base orientation
        perturb_orientation = self._episode_rng.uniform(
            low=-0.05 * np.pi, high=0.05 * np.pi
        )
        base_theta = -np.pi + theta + perturb_orientation

        # Torso height
        h = 0.5

        # Arm
        arm_qpos = [0, 0, 0, -1.5, 0, 3, 0.78, 0.02, 0.02]

        qpos = np.hstack([base_pos, base_theta, h, arm_qpos, arm_qpos])
        self.agent.reset(qpos)

    def _initialize_balls(self):
        bb = np.array(get_axis_aligned_bbox_for_articulation(self.bucket))
        R = self.balls_radius

        ball_idx = 0
        for i in range(self.GX):
            for j in range(self.GY):
                for k in range(self.GZ):
                    dx = -self.GX * R * 2 / 2 + R + 2 * R * i
                    dy = -self.GY * R * 2 / 2 + R + 2 * R * j
                    dz = R + R * 2 * k
                    pose = self.bucket.pose
                    pose = Pose(
                        [pose.p[0] + dx, pose.p[1] + dy, bb[1, 2] - bb[0, 2] + dz]
                    )
                    actor = self.balls[ball_idx]
                    actor.set_pose(pose)
                    ball_idx += 1

    def _set_bucket_links_pcd(self):
        for name, info in self.links_info.items():
            mesh: trimesh.Trimesh = info[1]
            with np_random(self._episode_seed):
                pcd = mesh.sample(512)
            self.links_info[name].append(pcd)

    # -------------------------------------------------------------------------- #
    # Success metric and shaped reward
    # -------------------------------------------------------------------------- #
    def evaluate(self, **kwargs):
        w2b = (
            self.bucket_body_link.pose.inv().to_transformation_matrix()
        )  # world to bucket

        in_bucket = True
        for b in self.balls:
            p = w2b[:3, :3] @ b.pose.p + w2b[:3, 3]
            if not np.all((p > self.bb_local[0]) * (p < self.bb_local[1])):
                in_bucket = False
                break

        z_axis_world = np.array([0, 0, 1])
        z_axis_bucket = quat2mat(self.root_link.get_pose().q) @ z_axis_world
        bucket_tilt = abs(angle_between_vec(z_axis_world, z_axis_bucket))

        dist_bucket_to_target = np.linalg.norm(
            self.root_link.get_pose().p[:2] - self.target_xy
        )

        vel_norm = np.linalg.norm(self.root_link.velocity)
        ang_vel_norm = np.linalg.norm(self.root_link.angular_velocity)

        flags = {
            "balls_in_bucket": in_bucket,
            "bucket_above_platform": dist_bucket_to_target < 0.3,
            "bucket_standing": bucket_tilt < 0.1 * np.pi,
            # "bucket_static": (vel_norm < 0.1 and ang_vel_norm < 0.2),
            "bucket_static": self.check_actor_static(
                self.root_link, max_v=0.1, max_ang_v=0.2
            ),
        }

        return dict(
            success=all(flags.values()),
            **flags,
            dist_bucket_to_target=dist_bucket_to_target,
            bucket_tilt=bucket_tilt,
            bucket_vel_norm=vel_norm,
            bucket_ang_vel_norm=ang_vel_norm,
        )

    def _get_bucket_pcd(self):
        """Get the point cloud of the bucket given its current joint positions."""
        links_pcd = []
        for name, info in self.links_info.items():
            link: sapien.LinkBase = info[0]
            pcd: np.ndarray = info[2]
            T = link.pose.to_transformation_matrix()
            pcd = transform_points(T, pcd)
            links_pcd.append(pcd)
        chair_pcd = np.concatenate(links_pcd, axis=0)
        return chair_pcd

    def compute_dense_reward(self, action, info: dict, **kwargs):
        reward = -20.0

        actor = self.root_link
        ee_coords = np.array(self.agent.get_ee_coords())
        ee_mids = np.array([ee_coords[:2].mean(0), ee_coords[2:].mean(0)])
        # ee_vels = np.array(self.agent.get_ee_vels())
        bucket_pcd = self._get_bucket_pcd()

        # EE approach bucket
        dist_ees_to_bucket = sdist.cdist(ee_coords, bucket_pcd)  # [M, N]
        dist_ees_to_bucket = dist_ees_to_bucket.min(1)  # [M]
        dist_ee_to_bucket = dist_ees_to_bucket.mean()
        log_dist_ee_to_bucket = np.log(dist_ee_to_bucket + 1e-5)
        reward += -dist_ee_to_bucket - np.clip(log_dist_ee_to_bucket, -10, 0)

        # EE adjust height
        bucket_mid = (
            self.bucket_body_link.get_pose()
            .transform(self.bucket_body_link.get_cmass_local_pose())
            .p
        )
        bucket_mid[2] += self.bucket_center_offset
        v1 = ee_mids[0] - bucket_mid
        v2 = ee_mids[1] - bucket_mid
        ees_oppo = sdist.cosine(v1, v2)
        ees_height_diff = abs(
            (quat2mat(self.root_link.get_pose().q).T @ (ee_mids[0] - ee_mids[1]))[2]
        )
        log_ees_height_diff = np.log(ees_height_diff + 1e-5)
        reward += -np.clip(log_ees_height_diff, -10, 0) * 0.2

        # Keep bucket standing
        bucket_tilt = info["bucket_tilt"]
        log_dist_ori = np.log(bucket_tilt + 1e-5)
        reward += -bucket_tilt * 0.2

        # Penalize action
        # Assume action is relative and normalized.
        action_norm = np.linalg.norm(action)
        reward -= action_norm * 1e-6

        # Bucket velocity
        actor_vel = actor.get_velocity()
        actor_vel_norm = np.linalg.norm(actor_vel)
        disp_bucket_to_target = self.root_link.get_pose().p[:2] - self.target_xy
        actor_vel_dir = sdist.cosine(actor_vel[:2], disp_bucket_to_target)
        actor_ang_vel_norm = np.linalg.norm(actor.get_angular_velocity())
        actor_vel_up = actor_vel[2]

        # Stage reward
        # NOTE(jigu): stage reward can also be used to debug which stage it is
        stage_reward = 0

        bucket_height = (
            self.bucket_body_link.get_pose()
            .transform(self.bucket_body_link.get_cmass_local_pose())
            .p[2]
        )
        dist_bucket_height = np.linalg.norm(
            bucket_height - self.init_bucket_height - 0.2
        )
        dist_bucket_to_target = info["dist_bucket_to_target"]

        if dist_ee_to_bucket < 0.1:
            stage_reward += 2
            reward += ees_oppo * 2  # - np.clip(log_ees_height_diff, -10, 0) * 0.2

            bucket_height = (
                self.bucket_body_link.get_pose()
                .transform(self.bucket_body_link.get_cmass_local_pose())
                .p[2]
            )
            dist_bucket_height = np.linalg.norm(
                bucket_height - self.init_bucket_height - 0.2
            )
            if dist_bucket_height < 0.03:
                stage_reward += 2
                reward -= np.clip(log_dist_ori, -4, 0)
                if dist_bucket_to_target <= 0.3:
                    stage_reward += 2
                    reward += (
                        np.exp(-actor_vel_norm * 10) * 2
                    )  # + np.exp(-actor_ang_vel_norm) * 0.5
                    if actor_vel_norm <= 0.1 and actor_ang_vel_norm <= 0.2:
                        stage_reward += 2
                        if bucket_tilt <= 0.1 * np.pi:
                            stage_reward += 2
                else:
                    reward_vel = (actor_vel_dir - 1) * actor_vel_norm
                    reward += (
                        np.clip(1 - np.exp(-reward_vel), -1, np.inf) * 2
                        - dist_bucket_to_target * 2
                    )
            else:
                reward += (
                    np.clip(1 - np.exp(-actor_vel_up), -1, np.inf) * 2
                    - dist_bucket_height * 20
                )

        if bucket_tilt > 0.4 * np.pi:
            stage_reward -= 2

        reward = reward + stage_reward

        # Update info
        info.update(
            action_norm=action_norm,
            stage_reward=stage_reward,
            dist_ee_to_bucket=dist_ee_to_bucket,
            bucket_height=bucket_height,
            ees_oppo=ees_oppo,
            ees_height_diff=ees_height_diff,
        )
        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 20.0

    # ---------------------------------------------------------------------------- #
    # Observation
    # ---------------------------------------------------------------------------- #
    def _get_task_actors(self):
        return self.balls

    def _get_task_articulations(self):
        # bucket max dof is 1 in our data
        return [(self.bucket, 2)]

    def set_state(self, state: np.ndarray):
        super().set_state(state)
        self._prev_actor_pose = self.bucket.pose

    def _get_obs_extra(self):
        obs = super()._get_obs_extra()
        if self._obs_mode not in ["state", "state_dict"]:
            obs["left_tcp_pose"] = vectorize_pose(self.left_tcp.pose)
            obs["right_tcp_pose"] = vectorize_pose(self.right_tcp.pose)
        return obs
