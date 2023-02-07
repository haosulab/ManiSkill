from collections import OrderedDict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qinverse, qmult, quat2axangle

from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import hex2rgba, look_at, vectorize_pose

from .base_env import StationaryManipulationEnv


@register_env("PlugCharger-v0", max_episode_steps=200)
class PlugChargerEnv(StationaryManipulationEnv):
    _base_size = [2e-2, 1.5e-2, 1.2e-2]  # charger base half size
    _peg_size = [8e-3, 0.75e-3, 3.2e-3]  # charger peg half size
    _peg_gap = 7e-3  # charger peg gap
    _clearance = 5e-4  # single side clearance
    _receptacle_size = [1e-2, 5e-2, 5e-2]  # receptacle half size

    def _build_charger(self, peg_size, base_size, gap):
        builder = self._scene.create_actor_builder()

        # peg
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#FFFFFF"))
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        builder.add_box_collision(Pose([peg_size[0], gap, 0]), peg_size)
        builder.add_box_visual(Pose([peg_size[0], gap, 0]), peg_size, material=mat)
        builder.add_box_collision(Pose([peg_size[0], -gap, 0]), peg_size)
        builder.add_box_visual(Pose([peg_size[0], -gap, 0]), peg_size, material=mat)

        # base
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#FFFFFF"))
        mat.metallic = 0.0
        mat.roughness = 0.1
        builder.add_box_collision(Pose([-base_size[0], 0, 0]), base_size)
        builder.add_box_visual(Pose([-base_size[0], 0, 0]), base_size, material=mat)

        return builder.build(name="charger")

    def _build_receptacle(self, peg_size, receptacle_size, gap):
        builder = self._scene.create_actor_builder()

        sy = 0.5 * (receptacle_size[1] - peg_size[1] - gap)
        sz = 0.5 * (receptacle_size[2] - peg_size[2])
        dx = -receptacle_size[0]
        dy = peg_size[1] + gap + sy
        dz = peg_size[2] + sz

        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#FFFFFF"))
        mat.metallic = 0.0
        mat.roughness = 0.1

        poses = [
            Pose([dx, 0, dz]),
            Pose([dx, 0, -dz]),
            Pose([dx, dy, 0]),
            Pose([dx, -dy, 0]),
        ]
        half_sizes = [
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], sy, receptacle_size[2]],
            [receptacle_size[0], sy, receptacle_size[2]],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=mat)

        # Fill the gap
        pose = Pose([-receptacle_size[0], 0, 0])
        half_size = [receptacle_size[0], gap - peg_size[1], peg_size[2]]
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)

        # Add dummy visual for hole
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#DBB539"))
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        pose = Pose([-receptacle_size[0], -(gap * 0.5 + peg_size[1]), 0])
        half_size = [receptacle_size[0], peg_size[1], peg_size[2]]
        builder.add_box_visual(pose, half_size, material=mat)
        pose = Pose([-receptacle_size[0], gap * 0.5 + peg_size[1], 0])
        builder.add_box_visual(pose, half_size, material=mat)

        return builder.build_static(name="receptacle")

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self.charger = self._build_charger(
            self._peg_size,
            self._base_size,
            self._peg_gap,
        )
        self.receptacle = self._build_receptacle(
            [
                self._peg_size[0],
                self._peg_size[1] + self._clearance,
                self._peg_size[2] + self._clearance,
            ],
            self._receptacle_size,
            self._peg_gap,
        )

    def _initialize_actors(self):
        # Initialize charger
        xy = self._episode_rng.uniform(
            [-0.1, -0.2], [-0.01 - self._peg_size[0] * 2, 0.2]
        )
        # xy = [-0.05, 0]
        pos = np.hstack([xy, self._base_size[2]])
        ori = self._episode_rng.uniform(-np.pi / 3, np.pi / 3)
        # ori = 0
        quat = euler2quat(0, 0, ori)
        self.charger.set_pose(Pose(pos, quat))

        # Initialize receptacle
        xy = self._episode_rng.uniform([0.01, -0.1], [0.1, 0.1])
        # xy = [0.05, 0]
        pos = np.hstack([xy, 0.1])
        ori = np.pi + self._episode_rng.uniform(-np.pi / 8, np.pi / 8)
        # ori = np.pi
        quat = euler2quat(0, 0, ori)
        self.receptacle.set_pose(Pose(pos, quat))

        # Adjust render camera
        if "render_camera" in self._render_cameras:
            self._render_cameras["render_camera"].camera.set_local_pose(
                self.receptacle.pose * look_at([0.3, 0.4, 0.1], [0, 0, 0])
            )

    def _initialize_task(self):
        self.goal_pose = self.receptacle.pose.transform(Pose(q=euler2quat(0, 0, np.pi)))
        # NOTE(jigu): clearance need to be set to 1e-3 so that the charger will not fall off
        # self.charger.set_pose(self.goal_pose)

    @property
    def charger_base_pose(self):
        return self.charger.pose.transform(Pose([-self._base_size[0], 0, 0]))

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(tcp_pose=vectorize_pose(self.tcp.pose))
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                charger_pose=vectorize_pose(self.charger.pose),
                receptacle_pose=vectorize_pose(self.receptacle.pose),
                goal_pose=vectorize_pose(self.goal_pose),
            )
        return obs

    def evaluate(self, **kwargs):
        obj_to_goal_dist, obj_to_goal_angle = self._compute_distance()
        success = obj_to_goal_dist <= 5e-3 and obj_to_goal_angle <= 0.2
        return dict(
            obj_to_goal_dist=obj_to_goal_dist,
            obj_to_goal_angle=obj_to_goal_angle,
            success=success,
        )

    def _compute_distance(self):
        obj_pose = self.charger.pose
        obj_to_goal_pos = self.goal_pose.p - obj_pose.p
        obj_to_goal_dist = np.linalg.norm(obj_to_goal_pos)

        obj_to_goal_quat = qmult(qinverse(self.goal_pose.q), obj_pose.q)
        _, obj_to_goal_angle = quat2axangle(obj_to_goal_quat)
        obj_to_goal_angle = min(obj_to_goal_angle, np.pi * 2 - obj_to_goal_angle)
        assert obj_to_goal_angle >= 0.0, obj_to_goal_angle

        return obj_to_goal_dist, obj_to_goal_angle

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            return 50.0

        cmass_pose = self.charger.pose.transform(self.charger.cmass_local_pose)
        # grasp pose rotation reward
        tcp_pose_wrt_charger = cmass_pose.inv() * self.tcp.pose
        tcp_rot_wrt_charger = tcp_pose_wrt_charger.to_transformation_matrix()[:3, :3]
        gt_rot_1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        gt_rot_2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        grasp_rot_loss_fxn = lambda A: np.tanh(
            1 / 4 * np.trace(A.T @ A)
        )  # trace(A.T @ A) has range [0,8] for A being difference of rotation matrices
        grasp_rot_loss = np.minimum(
            grasp_rot_loss_fxn(gt_rot_1 - tcp_rot_wrt_charger),
            grasp_rot_loss_fxn(gt_rot_2 - tcp_rot_wrt_charger),
        )
        reward += 2 * (1 - grasp_rot_loss)
        rotated_properly = grasp_rot_loss < 0.1

        if rotated_properly:
            # reaching reward
            gripper_to_obj_pos = cmass_pose.p - self.tcp.pose.p
            gripper_to_obj_dist = np.linalg.norm(gripper_to_obj_pos)
            reaching_reward = 1 - np.tanh(5.0 * gripper_to_obj_dist)
            reward += 2 * reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(
                self.charger, max_angle=20
            )  # max_angle ensures that the gripper grasps the charger appropriately, not in a strange pose
            if is_grasped:
                reward += 2.0

            # pre-insertion and insertion award
            if is_grasped:
                pre_inserted = False
                charger_cmass_wrt_goal = self.goal_pose.inv() * cmass_pose
                charger_cmass_wrt_goal_yz_dist = np.linalg.norm(
                    charger_cmass_wrt_goal.p[1:]
                )
                charger_cmass_wrt_goal_rot = (
                    charger_cmass_wrt_goal.to_transformation_matrix()[:3, :3]
                )
                charger_wrt_goal = self.goal_pose.inv() * self.charger.pose
                charger_wrt_goal_yz_dist = np.linalg.norm(charger_wrt_goal.p[1:])
                charger_wrt_goal_dist = np.linalg.norm(charger_wrt_goal.p)
                charger_wrt_goal_rot = (
                    charger_cmass_wrt_goal.to_transformation_matrix()[:3, :3]
                )

                gt_rot = np.eye(3)
                rot_loss_fxn = lambda A: np.tanh(1 / 2 * np.trace(A.T @ A))
                rot_loss = np.maximum(
                    rot_loss_fxn(charger_cmass_wrt_goal_rot - gt_rot),
                    rot_loss_fxn(charger_wrt_goal_rot - gt_rot),
                )

                pre_insertion_reward = 3 * (
                    1
                    - np.tanh(
                        1.0
                        * (charger_cmass_wrt_goal_yz_dist + charger_wrt_goal_yz_dist)
                        + 9.0
                        * np.maximum(
                            charger_cmass_wrt_goal_yz_dist, charger_wrt_goal_yz_dist
                        )
                    )
                )
                pre_insertion_reward += 3 * (1 - np.tanh(3 * charger_wrt_goal_dist))
                pre_insertion_reward += 3 * (1 - rot_loss)
                reward += pre_insertion_reward

                if (
                    charger_cmass_wrt_goal_yz_dist < 0.01
                    and charger_wrt_goal_yz_dist < 0.01
                    and charger_wrt_goal_dist < 0.02
                    and rot_loss < 0.15
                ):
                    pre_inserted = True
                    reward += 2.0

                if pre_inserted:
                    insertion_reward = 2 * (1 - np.tanh(25.0 * charger_wrt_goal_dist))
                    insertion_reward += 5 * (
                        1 - np.tanh(2.0 * np.abs(info["obj_to_goal_angle"]))
                    )
                    insertion_reward += 5 * (1 - rot_loss)
                    reward += insertion_reward
        else:
            reward = reward - 10 * np.maximum(
                self.charger.pose.p[2]
                + self._base_size[2] / 2
                + 0.015
                - self.tcp.pose.p[2],
                0.0,
            )
            reward = reward - 10 * np.linalg.norm(
                self.charger.pose.p[:2] - self.tcp.pose.p[:2]
            )

        return reward

    def _register_cameras(self):
        cam_cfg = super()._register_cameras()
        cam_cfg.pose = look_at([-0.3, 0, 0.1], [0, 0, 0.1])
        return cam_cfg

    def _register_render_cameras(self):
        cam_cfg = super()._register_render_cameras()
        cam_cfg.pose = look_at([-0.3, -0.4, 0.2], [0, 0, 0.1])
        return cam_cfg

    def _setup_lighting(self):
        super()._setup_lighting()
        if self.enable_shadow:
            self._scene.add_point_light(
                [-0.2, -0.5, 1], [2, 2, 2], shadow=self.enable_shadow
            )

    def set_state(self, state):
        super().set_state(state)
        self.goal_pose = self.receptacle.pose.transform(Pose(q=euler2quat(0, 0, np.pi)))
