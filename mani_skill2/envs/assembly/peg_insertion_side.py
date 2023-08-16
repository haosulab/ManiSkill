from collections import OrderedDict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat

from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import hex2rgba, look_at, vectorize_pose

from .base_env import StationaryManipulationEnv


@register_env("PegInsertionSide-v0", max_episode_steps=200)
class PegInsertionSideEnv(StationaryManipulationEnv):
    _clearance = 0.003

    def reset(self, seed=None, options=None):
        if options is None:
            options = {}
        if options.get("reconfigure") is None:
            options["reconfigure"] = True
        return super().reset(seed, options)

    def _build_box_with_hole(
        self, inner_radius, outer_radius, depth, center=(0, 0), name="box_with_hole"
    ):
        builder = self._scene.create_actor_builder()
        thickness = (outer_radius - inner_radius) * 0.5
        # x-axis is hole direction
        half_center = [x * 0.5 for x in center]
        half_sizes = [
            [depth, thickness - half_center[0], outer_radius],
            [depth, thickness + half_center[0], outer_radius],
            [depth, outer_radius, thickness - half_center[1]],
            [depth, outer_radius, thickness + half_center[1]],
        ]
        offset = thickness + inner_radius
        poses = [
            Pose([0, offset + half_center[0], 0]),
            Pose([0, -offset + half_center[0], 0]),
            Pose([0, 0, offset + half_center[1]]),
            Pose([0, 0, -offset + half_center[1]]),
        ]

        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#FFD289"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5

        for half_size, pose in zip(half_sizes, poses):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=mat)
        return builder.build_static(name)

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)

        # peg
        # length, radius = 0.1, 0.02
        length = self._episode_rng.uniform(0.075, 0.125)
        radius = self._episode_rng.uniform(0.015, 0.025)
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=[length, radius, radius])

        # peg head
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#EC7357"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5
        builder.add_box_visual(
            Pose([length / 2, 0, 0]),
            half_size=[length / 2, radius, radius],
            material=mat,
        )

        # peg tail
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#EDF6F9"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5
        builder.add_box_visual(
            Pose([-length / 2, 0, 0]),
            half_size=[length / 2, radius, radius],
            material=mat,
        )

        self.peg = builder.build("peg")
        self.peg_head_offset = Pose([length, 0, 0])
        self.peg_half_size = np.float32([length, radius, radius])

        # box with hole
        center = 0.5 * (length - radius) * self._episode_rng.uniform(-1, 1, size=2)
        inner_radius, outer_radius, depth = radius + self._clearance, length, length
        self.box = self._build_box_with_hole(
            inner_radius, outer_radius, depth, center=center
        )
        self.box_hole_offset = Pose(np.hstack([0, center]))
        self.box_hole_radius = inner_radius

    def _initialize_actors(self):
        xy = self._episode_rng.uniform([-0.1, -0.3], [0.1, 0])
        pos = np.hstack([xy, self.peg_half_size[2]])
        ori = np.pi / 2 + self._episode_rng.uniform(-np.pi / 3, np.pi / 3)
        quat = euler2quat(0, 0, ori)
        self.peg.set_pose(Pose(pos, quat))

        xy = self._episode_rng.uniform([-0.05, 0.2], [0.05, 0.4])
        pos = np.hstack([xy, self.peg_half_size[0]])
        ori = np.pi / 2 + self._episode_rng.uniform(-np.pi / 8, np.pi / 8)
        quat = euler2quat(0, 0, ori)
        self.box.set_pose(Pose(pos, quat))

    def _initialize_agent(self):
        if self.robot_uid == "panda":
            # fmt: off
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, -np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.615, 0, 0]))
        else:
            raise NotImplementedError(self.robot_uid)

    @property
    def peg_head_pos(self):
        return self.peg.pose.transform(self.peg_head_offset).p

    @property
    def peg_head_pose(self):
        return self.peg.pose.transform(self.peg_head_offset)

    @property
    def box_hole_pose(self):
        return self.box.pose.transform(self.box_hole_offset)

    def _initialize_task(self):
        self.goal_pos = self.box_hole_pose.p  # goal of peg head inside the hole
        # NOTE(jigu): The goal pose is computed based on specific geometries used in this task.
        # Only consider one side
        self.goal_pose = (
            self.box.pose * self.box_hole_offset * self.peg_head_offset.inv()
        )
        # self.peg.set_pose(self.goal_pose)

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(tcp_pose=vectorize_pose(self.tcp.pose))
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                peg_pose=vectorize_pose(self.peg.pose),
                peg_half_size=self.peg_half_size,
                box_hole_pose=vectorize_pose(self.box_hole_pose),
                box_hole_radius=self.box_hole_radius,
            )
        return obs

    def has_peg_inserted(self):
        # Only head position is used in fact
        peg_head_pose = self.peg.pose.transform(self.peg_head_offset)
        box_hole_pose = self.box_hole_pose
        peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p
        # x-axis is hole direction
        x_flag = -0.015 <= peg_head_pos_at_hole[0]
        y_flag = (
            -self.box_hole_radius <= peg_head_pos_at_hole[1] <= self.box_hole_radius
        )
        z_flag = (
            -self.box_hole_radius <= peg_head_pos_at_hole[2] <= self.box_hole_radius
        )
        return (x_flag and y_flag and z_flag), peg_head_pos_at_hole

    def evaluate(self, **kwargs) -> dict:
        success, peg_head_pos_at_hole = self.has_peg_inserted()
        return dict(success=success, peg_head_pos_at_hole=peg_head_pos_at_hole)

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            return 25.0

        # grasp pose rotation reward
        tcp_pose_wrt_peg = self.peg.pose.inv() * self.tcp.pose
        tcp_rot_wrt_peg = tcp_pose_wrt_peg.to_transformation_matrix()[:3, :3]
        gt_rot_1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        gt_rot_2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        grasp_rot_loss_fxn = lambda A: np.arcsin(
            np.clip(1 / (2 * np.sqrt(2)) * np.sqrt(np.trace(A.T @ A)), 0, 1)
        )
        grasp_rot_loss = np.minimum(
            grasp_rot_loss_fxn(gt_rot_1 - tcp_rot_wrt_peg),
            grasp_rot_loss_fxn(gt_rot_2 - tcp_rot_wrt_peg),
        ) / (np.pi / 2)
        rotated_properly = grasp_rot_loss < 0.2
        reward += 1 - grasp_rot_loss

        gripper_pos = self.tcp.pose.p
        tgt_gripper_pose = self.peg.pose
        offset = sapien.Pose(
            [-0.06, 0, 0]
        )  # account for panda gripper width with a bit more leeway
        tgt_gripper_pose = tgt_gripper_pose.transform(offset)
        if rotated_properly:
            # reaching reward
            gripper_to_peg_dist = np.linalg.norm(gripper_pos - tgt_gripper_pose.p)
            reaching_reward = 1 - np.tanh(
                4.0 * np.maximum(gripper_to_peg_dist - 0.015, 0.0)
            )
            # reaching_reward = 1 - np.tanh(10.0 * gripper_to_peg_dist)
            reward += reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(
                self.peg, max_angle=20
            )  # max_angle ensures that the gripper grasps the peg appropriately, not in a strange pose
            if is_grasped:
                reward += 2.0

            # pre-insertion award, encouraging both the peg center and the peg head to match the yz coordinates of goal_pose
            pre_inserted = False
            if is_grasped:
                peg_head_wrt_goal = self.goal_pose.inv() * self.peg_head_pose
                peg_head_wrt_goal_yz_dist = np.linalg.norm(peg_head_wrt_goal.p[1:])
                peg_wrt_goal = self.goal_pose.inv() * self.peg.pose
                peg_wrt_goal_yz_dist = np.linalg.norm(peg_wrt_goal.p[1:])
                if peg_head_wrt_goal_yz_dist < 0.01 and peg_wrt_goal_yz_dist < 0.01:
                    pre_inserted = True
                    reward += 3.0
                pre_insertion_reward = 3 * (
                    1
                    - np.tanh(
                        0.5 * (peg_head_wrt_goal_yz_dist + peg_wrt_goal_yz_dist)
                        + 4.5
                        * np.maximum(peg_head_wrt_goal_yz_dist, peg_wrt_goal_yz_dist)
                    )
                )
                reward += pre_insertion_reward

            # insertion reward
            if is_grasped and pre_inserted:
                peg_head_wrt_goal_inside_hole = (
                    self.box_hole_pose.inv() * self.peg_head_pose
                )
                insertion_reward = 5 * (
                    1 - np.tanh(5.0 * np.linalg.norm(peg_head_wrt_goal_inside_hole.p))
                )
                reward += insertion_reward
        else:
            reward = reward - 10 * np.maximum(
                self.peg.pose.p[2] + self.peg_half_size[2] + 0.01 - self.tcp.pose.p[2],
                0.0,
            )
            reward = reward - 10 * np.linalg.norm(
                tgt_gripper_pose.p[:2] - self.tcp.pose.p[:2]
            )

        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 25.0

    def _register_cameras(self):
        cam_cfg = super()._register_cameras()
        cam_cfg.pose = look_at([0, -0.3, 0.2], [0, 0, 0.1])
        return cam_cfg

    def _register_render_cameras(self):
        cam_cfg = super()._register_render_cameras()
        cam_cfg.pose = look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])
        return cam_cfg

    def set_state(self, state):
        super().set_state(state)
        # NOTE(xuanlin): This way is specific to how we compute goals.
        # The general way is to handle variables explicitly
        self._initialize_task()
