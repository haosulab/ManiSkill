from collections import OrderedDict

import numpy as np
import sapien.core as sapien
import trimesh
from sapien.core import Pose
from scipy.spatial import distance as sdist

from mani_skill2.agents.robots.mobile_panda import MobilePandaSingleArm
from mani_skill2.utils.common import np_random, random_choice
from mani_skill2.utils.geometry import angle_distance, transform_points
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import get_entity_by_name, vectorize_pose
from mani_skill2.utils.trimesh_utils import (
    get_articulation_meshes,
    get_visual_body_meshes,
    merge_meshes,
)

from .base_env import MS1BaseEnv


def clip_and_normalize(x, a_min, a_max=None):
    if a_max is None:
        a_max = np.abs(a_min)
        a_min = -a_max
    return (np.clip(x, a_min, a_max) - a_min) / (a_max - a_min)


class OpenCabinetEnv(MS1BaseEnv):
    ASSET_UID = "partnet_mobility_cabinet"
    MAX_DOF = 8
    agent: MobilePandaSingleArm

    def __init__(self, *args, fixed_target_link_idx: int = None, **kwargs):
        # The index in target links (not all links)
        self._fixed_target_link_idx = fixed_target_link_idx
        self._cache_bboxes = {}
        super().__init__(*args, **kwargs)

    def _register_render_cameras(self):
        cam_cfg = super()._register_render_cameras()
        cam_cfg.p = [-1.5, 0, 1.5]
        cam_cfg.q = [0.9238795, 0, 0.3826834, 0]
        return cam_cfg

    # -------------------------------------------------------------------------- #
    # Reconfigure
    # -------------------------------------------------------------------------- #
    def _load_articulations(self):
        urdf_config = dict(
            material=dict(static_friction=1, dynamic_friction=1, restitution=0),
        )
        scale = self.model_info["scale"]
        self.cabinet = self._load_partnet_mobility(
            fix_root_link=True, scale=scale, urdf_config=urdf_config
        )
        self.cabinet.set_name(self.model_id)

        assert self.cabinet.dof <= self.MAX_DOF, self.cabinet.dof
        self._set_cabinet_handles()
        self._ignore_collision()

        if self._reward_mode in ["dense", "normalized_dense"]:
            # NOTE(jigu): Explicit `set_pose` is needed.
            self.cabinet.set_pose(Pose())
            self._set_cabinet_handles_mesh()
            self._compute_handles_grasp_poses()

    def _set_cabinet_handles(self, joint_type: str):
        self.target_links = []
        self.target_joints = []
        self.target_handles = []

        # NOTE(jigu): links and their parent joints.
        for link, joint in zip(self.cabinet.get_links(), self.cabinet.get_joints()):
            if joint.type != joint_type:
                continue
            handles = []
            for visual_body in link.get_visual_bodies():
                if "handle" not in visual_body.name:
                    continue
                handles.append(visual_body)
            if len(handles) > 0:
                self.target_links.append(link)
                self.target_joints.append(joint)
                self.target_handles.append(handles)

    def _set_cabinet_handles_mesh(self):
        self.target_handles_mesh = []

        for handle_visuals in self.target_handles:
            meshes = []
            for visual_body in handle_visuals:
                meshes.extend(get_visual_body_meshes(visual_body))
            handle_mesh = merge_meshes(meshes)
            # Legacy issue: convex hull is assumed for further computation
            handle_mesh = trimesh.convex.convex_hull(handle_mesh)
            self.target_handles_mesh.append(handle_mesh)

    def _compute_grasp_poses(self, mesh: trimesh.Trimesh, pose: sapien.Pose):
        # NOTE(jigu): only for axis-aligned horizontal and vertical cases
        mesh2: trimesh.Trimesh = mesh.copy()
        # Assume the cabinet is axis-aligned canonically
        mesh2.apply_transform(pose.to_transformation_matrix())

        extents = mesh2.extents
        if extents[1] > extents[2]:  # horizontal handle
            closing = np.array([0, 0, 1])
        else:  # vertical handle
            closing = np.array([0, 1, 0])

        # Only rotation of grasp poses are used. Thus, center is dummy.
        approaching = [1, 0, 0]
        grasp_poses = [
            self.agent.build_grasp_pose(approaching, closing, [0, 0, 0]),
            self.agent.build_grasp_pose(approaching, -closing, [0, 0, 0]),
        ]

        # # Visualization
        # grasp_T = grasp_poses[0].to_transformation_matrix()
        # coord_frame = trimesh.creation.axis(
        #     transform=grasp_T, origin_size=0.001, axis_radius=0.001, axis_length=0.01
        # )
        # trimesh.Scene([mesh2, coord_frame]).show()

        pose_inv = pose.inv()
        grasp_poses = [pose_inv * x for x in grasp_poses]

        return grasp_poses

    def _compute_handles_grasp_poses(self):
        self.target_handles_grasp_poses = []
        for i in range(len(self.target_handles)):
            link = self.target_links[i]
            mesh = self.target_handles_mesh[i]
            grasp_poses = self._compute_grasp_poses(mesh, link.pose)
            self.target_handles_grasp_poses.append(grasp_poses)

    def _ignore_collision(self):
        """Ignore collision within the articulation to avoid impact from imperfect collision shapes."""
        # The legacy version only ignores collision of child links of active joints.
        for link in self.cabinet.get_links():
            for s in link.get_collision_shapes():
                g0, g1, g2, g3 = s.get_collision_groups()
                s.set_collision_groups(g0, g1, g2 | 1 << 31, g3)

    def _configure_agent(self):
        self._agent_cfg = MobilePandaSingleArm.get_default_config()

    def _load_agent(self):
        self.agent = MobilePandaSingleArm(
            self._scene, self._control_freq, self._control_mode, config=self._agent_cfg
        )

        links = self.agent.robot.get_links()
        self.tcp: sapien.Link = get_entity_by_name(links, "right_panda_hand_tcp")

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        return super().reset(seed=seed, options=options)

    def _initialize_task(self):
        self._initialize_cabinet()
        self._initialize_robot()
        self._set_target_link()
        self._set_joint_physical_parameters()

    def _compute_cabinet_bbox(self):
        mesh = merge_meshes(get_articulation_meshes(self.cabinet))
        return mesh.bounds  # [2, 3]

    def _initialize_cabinet(self):
        # Set joint positions to lower bounds
        qlimits = self.cabinet.get_qlimits()  # [N, 2]
        assert not np.isinf(qlimits).any(), qlimits
        qpos = np.ascontiguousarray(qlimits[:, 0])
        # NOTE(jigu): must use a contiguous array for `set_qpos`
        self.cabinet.set_qpos(qpos)

        # If the scale can change, caching does not work.
        bounds = self._cache_bboxes.get(self.model_id, None)
        if bounds is None:
            # The bound is computed based on current qpos.
            # NOTE(jigu): Make sure the box is computed at a canoncial pose.
            self.cabinet.set_pose(Pose())
            bounds = self._compute_cabinet_bbox()
            self._cache_bboxes[self.model_id] = bounds
        self.cabinet.set_pose(Pose([0, 0, -bounds[0, 2]]))

    def _initialize_robot(self):
        # Base position
        # The forward direction of cabinets is -x.
        center = np.array([0, 0.8])
        dist = self._episode_rng.uniform(1.6, 1.8)
        theta = self._episode_rng.uniform(0.9 * np.pi, 1.1 * np.pi)
        direction = np.array([np.cos(theta), np.sin(theta)])
        xy = center + direction * dist

        # Base orientation
        noise_ori = self._episode_rng.uniform(-0.05 * np.pi, 0.05 * np.pi)
        ori = (theta - np.pi) + noise_ori

        h = 1e-4
        arm_qpos = np.array([0, 0, 0, -1.5, 0, 3, 0.78, 0.02, 0.02])

        qpos = np.hstack([xy, ori, h, arm_qpos])
        self.agent.reset(qpos)

    def _set_joint_physical_parameters(self):
        for joint in self.cabinet.get_active_joints():
            joint.set_friction(self._episode_rng.uniform(0.05, 0.15))
            joint.set_drive_property(
                stiffness=0, damping=self._episode_rng.uniform(5, 20)
            )

    def _set_target_link(self):
        if self._fixed_target_link_idx is None:
            indices = np.arange(len(self.target_links))
            self.target_link_idx = random_choice(indices, rng=self._episode_rng)
        else:
            self.target_link_idx = self._fixed_target_link_idx
        assert self.target_link_idx < len(self.target_links), self.target_link_idx

        self.target_link: sapien.Link = self.target_links[self.target_link_idx]
        self.target_joint: sapien.Joint = self.target_joints[self.target_link_idx]
        # The index in active joints
        self.target_joint_idx_q = self.cabinet.get_active_joints().index(
            self.target_joint
        )

        qmin, qmax = self.target_joint.get_limits()[0]
        self.target_qpos = qmin + (qmax - qmin) * 0.9
        self.target_angle_diff = self.target_qpos - qmin

        # One-hot indicator for which link is target
        self.target_indicator = np.zeros(self.MAX_DOF, np.float32)
        self.target_indicator[self.target_joint_idx_q] = 1

        # x-axis is the revolute/prismatic joint direction
        joint_pose = self.target_joint.get_global_pose().to_transformation_matrix()
        self.target_joint_axis = joint_pose[:3, 0]
        # It is not handle position
        cmass_pose = self.target_link.pose * self.target_link.cmass_local_pose
        self.target_link_pos = cmass_pose.p

        # Cache handle point cloud
        if self._reward_mode in ["dense", "normalized_dense"]:
            self._set_target_handle_info()

    def _set_target_handle_info(self):
        self.target_handle_mesh = self.target_handles_mesh[self.target_link_idx]
        with np_random(self._episode_seed):
            self.target_handle_pcd = self.target_handle_mesh.sample(100)
        self.target_handle_sdf = trimesh.proximity.ProximityQuery(
            self.target_handle_mesh
        )

    # -------------------------------------------------------------------------- #
    # Success metric and shaped reward
    # -------------------------------------------------------------------------- #
    @property
    def link_qpos(self):
        return self.cabinet.get_qpos()[self.target_joint_idx_q]

    @property
    def link_qvel(self):
        return self.cabinet.get_qvel()[self.target_joint_idx_q]

    def evaluate(self, **kwargs) -> dict:
        vel_norm = np.linalg.norm(self.target_link.velocity)
        ang_vel_norm = np.linalg.norm(self.target_link.angular_velocity)
        link_qpos = self.link_qpos

        flags = dict(
            # cabinet_static=vel_norm <= 0.1 and ang_vel_norm <= 1,
            cabinet_static=self.check_actor_static(
                self.target_link, max_v=0.1, max_ang_v=1
            ),
            open_enough=link_qpos >= self.target_qpos,
        )

        return dict(
            success=all(flags.values()),
            **flags,
            link_vel_norm=vel_norm,
            link_ang_vel_norm=ang_vel_norm,
            link_qpos=link_qpos
        )

    def compute_dense_reward(self, *args, info: dict, **kwargs):
        reward = 0.0

        # -------------------------------------------------------------------------- #
        # The end-effector should be close to the target pose
        # -------------------------------------------------------------------------- #
        handle_pose = self.target_link.pose
        ee_pose = self.agent.hand.pose

        # Position
        ee_coords = self.agent.get_ee_coords_sample()  # [2, 10, 3]
        handle_pcd = transform_points(
            handle_pose.to_transformation_matrix(), self.target_handle_pcd
        )
        # trimesh.PointCloud(handle_pcd).show()
        disp_ee_to_handle = sdist.cdist(ee_coords.reshape(-1, 3), handle_pcd)
        dist_ee_to_handle = disp_ee_to_handle.reshape(2, -1).min(-1)  # [2]
        reward_ee_to_handle = -dist_ee_to_handle.mean() * 2
        reward += reward_ee_to_handle

        # Encourage grasping the handle
        ee_center_at_world = ee_coords.mean(0)  # [10, 3]
        ee_center_at_handle = transform_points(
            handle_pose.inv().to_transformation_matrix(), ee_center_at_world
        )
        # self.ee_center_at_handle = ee_center_at_handle
        dist_ee_center_to_handle = self.target_handle_sdf.signed_distance(
            ee_center_at_handle
        )
        # print("SDF", dist_ee_center_to_handle)
        dist_ee_center_to_handle = dist_ee_center_to_handle.max()
        reward_ee_center_to_handle = (
            clip_and_normalize(dist_ee_center_to_handle, -0.01, 4e-3) - 1
        )
        reward += reward_ee_center_to_handle

        # pointer = trimesh.creation.icosphere(radius=0.02, color=(1, 0, 0))
        # trimesh.Scene([self.target_handle_mesh, trimesh.PointCloud(ee_center_at_handle)]).show()

        # Rotation
        target_grasp_poses = self.target_handles_grasp_poses[self.target_link_idx]
        target_grasp_poses = [handle_pose * x for x in target_grasp_poses]
        angles_ee_to_grasp_poses = [
            angle_distance(ee_pose, x) for x in target_grasp_poses
        ]
        ee_rot_reward = -min(angles_ee_to_grasp_poses) / np.pi * 3
        reward += ee_rot_reward

        # -------------------------------------------------------------------------- #
        # Stage reward
        # -------------------------------------------------------------------------- #
        coeff_qvel = 1.5  # joint velocity
        coeff_qpos = 0.5  # joint position distance
        stage_reward = -5 - (coeff_qvel + coeff_qpos)
        # Legacy version also abstract coeff_qvel + coeff_qpos.

        link_qpos = info["link_qpos"]
        link_qvel = self.link_qvel
        link_vel_norm = info["link_vel_norm"]
        link_ang_vel_norm = info["link_ang_vel_norm"]

        ee_close_to_handle = (
            dist_ee_to_handle.max() <= 0.01 and dist_ee_center_to_handle > 0
        )
        if ee_close_to_handle:
            stage_reward += 0.5

            # Distance between current and target joint positions
            # TODO(jigu): the lower bound 0 is problematic? should we use lower bound of joint limits?
            reward_qpos = (
                clip_and_normalize(link_qpos, 0, self.target_qpos) * coeff_qpos
            )
            reward += reward_qpos

            if not info["open_enough"]:
                # Encourage positive joint velocity to increase joint position
                reward_qvel = clip_and_normalize(link_qvel, -0.1, 0.5) * coeff_qvel
                reward += reward_qvel
            else:
                # Add coeff_qvel for smooth transition of stagess
                stage_reward += 2 + coeff_qvel
                reward_static = -(link_vel_norm + link_ang_vel_norm * 0.5)
                reward += reward_static

                # Legacy version uses static from info, which is incompatible with MPC.
                # if info["cabinet_static"]:
                if link_vel_norm <= 0.1 and link_ang_vel_norm <= 1:
                    stage_reward += 1

        # Update info
        info.update(ee_close_to_handle=ee_close_to_handle, stage_reward=stage_reward)

        reward += stage_reward
        return reward

    def compute_normalized_dense_reward(self, *args, info: dict, **kwargs):
        return self.compute_dense_reward(*args, info=info, **kwargs) / 10.0

    # -------------------------------------------------------------------------- #
    # Observations
    # -------------------------------------------------------------------------- #
    def _get_obs_extra(self) -> OrderedDict:
        obs = super()._get_obs_extra()
        if self._obs_mode not in ["state", "state_dict"]:
            obs.update(
                target_angle_diff=self.target_angle_diff,
                target_joint_axis=self.target_joint_axis,
                target_link_pos=self.target_link_pos,
            )
            obs["tcp_pose"] = vectorize_pose(self.tcp.pose)
        return obs

    def _get_obs_priviledged(self):
        obs = super()._get_obs_priviledged()
        obs["target_indicator"] = self.target_indicator
        return obs

    def _get_task_articulations(self):
        # The maximum DoF is 6 in our data.
        return [(self.cabinet, 8)]

    def set_state(self, state: np.ndarray):
        super().set_state(state)
        self._prev_actor_pose = self.target_link.pose


@register_env("OpenCabinetDoor-v1", max_episode_steps=200)
class OpenCabinetDoorEnv(OpenCabinetEnv):
    DEFAULT_MODEL_JSON = (
        "{PACKAGE_ASSET_DIR}/partnet_mobility/meta/info_cabinet_door_train.json"
    )

    def _set_cabinet_handles(self):
        super()._set_cabinet_handles("revolute")


@register_env("OpenCabinetDrawer-v1", max_episode_steps=200)
class OpenCabinetDrawerEnv(OpenCabinetEnv):
    DEFAULT_MODEL_JSON = (
        "{PACKAGE_ASSET_DIR}/partnet_mobility/meta/info_cabinet_drawer_train.json"
    )

    def _set_cabinet_handles(self):
        super()._set_cabinet_handles("prismatic")
