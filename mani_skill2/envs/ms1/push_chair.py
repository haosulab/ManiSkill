import numpy as np
import sapien.core as sapien
import trimesh
from sapien.core import Pose
from scipy.spatial import distance as sdist
from transforms3d.euler import euler2quat, quat2euler

from mani_skill2.agents.robots.mobile_panda import MobilePandaDualArm
from mani_skill2.utils.common import np_random
from mani_skill2.utils.geometry import transform_points
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import get_entity_by_name, vectorize_pose
from mani_skill2.utils.trimesh_utils import get_actor_visual_mesh

from .base_env import MS1BaseEnv


@register_env("PushChair-v1", max_episode_steps=200)
class PushChairEnv(MS1BaseEnv):
    DEFAULT_MODEL_JSON = (
        "{PACKAGE_ASSET_DIR}/partnet_mobility/meta/info_chair_train.json"
    )
    ASSET_UID = "partnet_mobility_chair"

    def _get_default_scene_config(self):
        scene_config = super()._get_default_scene_config()
        scene_config.solver_iterations = 15
        return scene_config

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
            material=dict(static_friction=0.1, dynamic_friction=0.1, restitution=0),
            density=200,
        )
        self.chair = self._load_partnet_mobility(
            fix_root_link=False, scale=0.8, urdf_config=urdf_config
        )
        self.chair.set_name(self.model_id)

        self._set_chair_links()
        self._ignore_collision()

        if self._reward_mode in ["dense", "normalized_dense"]:
            self._set_chair_links_mesh()

    @staticmethod
    def _check_link_types(link: sapien.LinkBase):
        link_types = []
        for visual_body in link.get_visual_bodies():
            name = visual_body.name
            if "wheel" in name:
                link_types.append("wheel")
            if "seat" in name:
                link_types.append("seat")
            if "leg" in name or "foot" in name:
                link_types.append("support")
        return link_types

    def _set_chair_links(self):
        chair_links = self.chair.get_links()

        # Infer link types
        self.root_link = chair_links[0]
        self.wheel_links = []
        self.seat_link = None
        self.support_link = None
        for link in chair_links:
            link_types = self._check_link_types(link)
            if "wheel" in link_types:
                self.wheel_links.append(link)
            if "seat" in link_types:
                assert self.seat_link is None, (self.seat_link, link)
                self.seat_link = link
            if "support" in link_types:
                assert self.support_link is None, (self.support_link, link)
                self.support_link = link

        # Set the physical material for wheels
        wheel_material = self._scene.create_physical_material(
            static_friction=1, dynamic_friction=1, restitution=0
        )
        for link in self.wheel_links:
            for s in link.get_collision_shapes():
                s.set_physical_material(wheel_material)

    def _ignore_collision(self):
        """Ignore collision within the articulation to avoid impact from imperfect collision shapes."""
        if self.seat_link == self.support_link:  # sometimes they are the same one
            return

        for link in [self.seat_link, self.support_link]:
            shapes = link.get_collision_shapes()
            for s in shapes:
                g0, g1, g2, g3 = s.get_collision_groups()
                s.set_collision_groups(g0, g1, g2 | 1 << 31, g3)

    def _load_actors(self):
        super()._load_actors()

        # A red sphere to indicate the target to push the chair.
        builder = self._scene.create_actor_builder()
        builder.add_sphere_visual(radius=0.15, color=(1, 0, 0))
        self.target_indicator = builder.build_static(name="target_indicator")

    def _configure_agent(self):
        self._agent_cfg = MobilePandaDualArm.get_default_config()
        self._agent_cfg.camera_h = 2

    def _load_agent(self):
        self.agent = MobilePandaDualArm(
            self._scene, self._control_freq, self._control_mode, config=self._agent_cfg
        )

        links = self.agent.robot.get_links()
        self.left_tcp: sapien.Link = get_entity_by_name(links, "left_panda_hand_tcp")
        self.right_tcp: sapien.Link = get_entity_by_name(links, "right_panda_hand_tcp")

    def _set_chair_links_mesh(self):
        self.links_info = {}
        for link in self.chair.get_links():
            mesh = get_actor_visual_mesh(link)
            if mesh is None:
                continue
            self.links_info[link.name] = [link, mesh]

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    def _initialize_task(self):
        self._set_target()
        # NOTE(jigu): Initialize articulations and agent after the target is determined.
        self._initialize_chair()
        self._initialize_robot()

        if self._reward_mode in ["dense", "normalized_dense"]:
            self._set_chair_links_pcd()
        self._set_joint_physical_parameters()

    def _set_target(self):
        self.target_xy = np.zeros(2, dtype=np.float32)
        self.target_p = np.zeros(3, dtype=np.float32)
        self.target_p[:2] = self.target_xy
        self.target_indicator.set_pose(Pose(p=self.target_p))

    def _initialize_chair(self):
        # Load resting states from model info
        p = np.float32(self.model_info["position"])
        q = np.float32(self.model_info["rotation"])
        init_qpos = self.model_info["initial_qpos"]
        self.chair.set_pose(Pose(p, q))
        self.chair.set_qpos(init_qpos)

        # Sample a position around the target
        r = self._episode_rng.uniform(0.8, 1.2)
        theta = self._episode_rng.uniform(-np.pi, np.pi)
        p[0] = np.cos(theta) * r
        p[1] = np.sin(theta) * r

        # The chair faces towards the origin.
        chair_init_ori = theta - np.pi
        # Add a perturbation
        noise_ori = self._episode_rng.uniform(-0.4 * np.pi, 0.4 * np.pi)
        chair_init_ori = chair_init_ori + noise_ori
        self.chair_init_ori = chair_init_ori

        # NOTE(jigu): The base link may not be axis-aligned,
        # while the seat link is assumed to be aligned (z forward and y up).
        # seat_dir = self.seat_link.pose.to_transformation_matrix()[:3, 2]
        # seat_ori = np.arctan2(seat_dir[1], seat_dir[0])
        # # Compute the relative angle from seat to chair orientation
        # delta_ori = self.chair_init_ori - seat_ori
        # ax, ay, az = quat2euler(q, "sxyz")
        # az = az + delta_ori
        # q = euler2quat(ax, ay, az, "sxyz")

        # Legacy way to compute orientation (less readable)
        _, _, az_seat = quat2euler(self.seat_link.get_pose().q, "sxyz")
        ax, ay, az = quat2euler(q, "sxyz")
        az = az - az_seat + np.pi * 1.5 + theta  # face to target
        az = az + noise_ori
        q = euler2quat(ax, ay, az, "sxyz")

        # Finalize chair pose
        self.chair.set_pose(Pose(p, q))

    def _initialize_robot(self):
        # Base position (around the chair, away from target)
        xy = self.chair.pose.p[:2]
        r = self._episode_rng.uniform(0.8, 1.2)
        theta = self._episode_rng.uniform(-0.2 * np.pi, 0.2 * np.pi)
        theta = (self.chair_init_ori + np.pi) + theta
        xy[0] += np.cos(theta) * r
        xy[1] += np.sin(theta) * r

        # Base orientation
        noise_ori = self._episode_rng.uniform(-0.05 * np.pi, 0.05 * np.pi)
        ori = (theta - np.pi) + noise_ori

        # Torso height
        h = 0.9

        # Arm
        arm_qpos = [0, 0, 0, -1.5, 0, 3, 0.78, 0.02, 0.02]

        qpos = np.hstack([xy, ori, h, arm_qpos, arm_qpos])
        self.agent.reset(qpos)

    def _set_joint_physical_parameters(self):
        for joint in self.chair.get_active_joints():
            parent_link = joint.get_parent_link()

            # revolute joint between seat and support
            if parent_link is not None and "helper" in parent_link.name:
                # assert joint.type == "revolute", (self.model_id, joint.type)
                joint.set_friction(self._episode_rng.uniform(0.05, 0.15))
                joint.set_drive_property(
                    stiffness=0, damping=self._episode_rng.uniform(5, 15)
                )
            else:
                joint.set_friction(self._episode_rng.uniform(0.0, 0.1))
                joint.set_drive_property(
                    stiffness=0, damping=self._episode_rng.uniform(0, 0.5)
                )

    def _set_chair_links_pcd(self):
        for name, info in self.links_info.items():
            mesh: trimesh.Trimesh = info[1]
            with np_random(self._episode_seed):
                pcd = mesh.sample(512)
            self.links_info[name].append(pcd)

    # -------------------------------------------------------------------------- #
    # Success metric and shaped reward
    # -------------------------------------------------------------------------- #
    def evaluate(self, **kwargs):
        disp_chair_to_target = self.chair.pose.p[:2] - self.target_xy
        dist_chair_to_target = np.linalg.norm(disp_chair_to_target)

        # z-axis of chair should be upward
        z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
        chair_tilt = np.arccos(z_axis_chair[2])

        vel_norm = np.linalg.norm(self.root_link.velocity)
        ang_vel_norm = np.linalg.norm(self.root_link.angular_velocity)

        flags = dict(
            chair_close_to_target=dist_chair_to_target < 0.15,
            chair_standing=chair_tilt < 0.05 * np.pi,
            # chair_static=(vel_norm < 0.1 and ang_vel_norm < 0.2),
            chair_static=self.check_actor_static(
                self.root_link, max_v=0.1, max_ang_v=0.2
            ),
        )
        return dict(
            success=all(flags.values()),
            **flags,
            dist_chair_to_target=dist_chair_to_target,
            chair_tilt=chair_tilt,
            chair_vel_norm=vel_norm,
            chair_ang_vel_norm=ang_vel_norm,
        )

    def _get_chair_pcd(self):
        """Get the point cloud of the chair given its current joint positions."""
        links_pcd = []
        for name, info in self.links_info.items():
            link: sapien.LinkBase = info[0]
            pcd: np.ndarray = info[2]
            T = link.pose.to_transformation_matrix()
            pcd = transform_points(T, pcd)
            links_pcd.append(pcd)
        chair_pcd = np.concatenate(links_pcd, axis=0)
        return chair_pcd

    def compute_dense_reward(self, action: np.ndarray, info: dict, **kwargs):
        reward = 0

        # Compute distance between end-effectors and chair surface
        ee_coords = np.array(self.agent.get_ee_coords())  # [M, 3]
        chair_pcd = self._get_chair_pcd()  # [N, 3]

        # EE approach chair
        dist_ees_to_chair = sdist.cdist(ee_coords, chair_pcd)  # [M, N]
        dist_ees_to_chair = dist_ees_to_chair.min(1)  # [M]
        dist_ee_to_chair = dist_ees_to_chair.mean()
        log_dist_ee_to_chair = np.log(dist_ee_to_chair + 1e-5)
        reward += -dist_ee_to_chair - np.clip(log_dist_ee_to_chair, -10, 0)

        # Keep chair standing
        chair_tilt = info["chair_tilt"]
        reward += -chair_tilt * 0.2

        # Penalize action
        # Assume action is relative and normalized.
        action_norm = np.linalg.norm(action)
        reward -= action_norm * 1e-6

        # Chair velocity
        # Legacy version uses full velocity instead of xy-plane velocity
        chair_vel = self.root_link.velocity[:2]
        chair_vel_norm = np.linalg.norm(chair_vel)
        disp_chair_to_target = self.root_link.get_pose().p[:2] - self.target_xy
        cos_chair_vel_to_target = sdist.cosine(disp_chair_to_target, chair_vel)
        chair_ang_vel_norm = info["chair_ang_vel_norm"]

        # Stage reward
        # NOTE(jigu): stage reward can also be used to debug which stage it is
        stage_reward = -10
        # -18 can guarantee the reward is negative
        dist_chair_to_target = info["dist_chair_to_target"]

        if chair_tilt < 0.2 * np.pi:
            # Chair is standing
            if dist_ee_to_chair < 0.1:
                # EE is close to chair
                stage_reward += 2
                if dist_chair_to_target <= 0.15:
                    # Chair is close to target
                    stage_reward += 2
                    # Try to keep chair static
                    reward += np.exp(-chair_vel_norm * 10) * 2
                    # Legacy: Note that the static condition here is different from success metric
                    if chair_vel_norm <= 0.1 and chair_ang_vel_norm <= 0.2:
                        stage_reward += 2
                else:
                    # Try to increase velocity along direction to the target
                    # Compute directional velocity
                    x = (1 - cos_chair_vel_to_target) * chair_vel_norm
                    reward += max(-1, 1 - np.exp(x)) * 2 - dist_chair_to_target * 2
        else:
            stage_reward = -5

        reward = reward + stage_reward

        # Update info
        info.update(
            dist_ee_to_chair=dist_ee_to_chair,
            action_norm=action_norm,
            chair_vel_norm=chair_vel_norm,
            cos_chair_vel_to_target=cos_chair_vel_to_target,
            stage_reward=stage_reward,
        )
        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 10.0

    # ---------------------------------------------------------------------------- #
    # Observations
    # ---------------------------------------------------------------------------- #
    def _get_task_articulations(self):
        # The maximum DoF is 20 in our data.
        return [(self.chair, 25)]

    def set_state(self, state: np.ndarray):
        super().set_state(state)
        self._prev_actor_pose = self.root_link.pose

    def _get_obs_extra(self):
        obs = super()._get_obs_extra()
        if self._obs_mode not in ["state", "state_dict"]:
            obs["left_tcp_pose"] = vectorize_pose(self.left_tcp.pose)
            obs["right_tcp_pose"] = vectorize_pose(self.right_tcp.pose)
        return obs
