from collections import OrderedDict
import time

import numpy as np
from gym import spaces
import pyrealsense2 as rs
from sapien.core import Pose
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import axangle2quat

from xarm.wrapper import XArmAPI
from real_robot.utils.logger import get_logger
from real_robot.utils.common import clip_and_scale_action, vectorize_pose
from real_robot.sensors.camera import CameraConfig


class XArm7:
    """
    xArm7 agent class
    Mimics mani_skill2.agents.base_agent.BaseAgent interface
    """
    SUPPORTED_CONTROL_MODES = ("pd_ee_pos", "pd_ee_delta_pos",
                               "pd_ee_pose_axangle", "pd_ee_delta_pose_axangle",
                               "pd_ee_pose_quat", "pd_ee_delta_pose_quat")
    SUPPORTED_MOTION_MODES = ("position", "servo",
                              "joint_teaching", "cartesian_teaching (invalid)",
                              "joint_vel", "cartesian_vel",
                              "joint_online", "cartesian_online")

    def __init__(self,
                 ip="192.168.1.229",
                 control_mode="pd_ee_delta_pos",
                 motion_mode="position",
                 safety_boundary=[999, -999, 999, -999, 999, 0],
                 boundary_clip_eps=10,
                 with_hand_camera=True):
        """
        :param motion_mode: xArm motion mode
        :param safety_boundary: [x_max, x_min, y_max, y_min, z_max, z_min] (mm)
        :param boundary_clip_eps: clip action when TCP position to boundary is
                                  within boundary_clip_eps (mm)
        :param with_hand_camera: whether to include hand camera mount in TCP offset.
        """
        self.logger = get_logger("XArm7")
        self.arm = XArmAPI(ip)

        if control_mode not in self.SUPPORTED_CONTROL_MODES:
            raise NotImplementedError(f"Unsupported {control_mode = }")
        if motion_mode not in self.SUPPORTED_MOTION_MODES:
            raise NotImplementedError(f"Unsupported {motion_mode = }")
        self._control_mode = control_mode
        self._motion_mode = motion_mode

        # TODO: read this from URDF
        self.joint_limits_ms2 = np.array(
            [[-6.2831855, 6.2831855],
             [-2.059, 2.0944],
             [-6.2831855, 6.2831855],
             [-0.19198, 3.927],
             [-6.2831855, 6.2831855],
             [-1.69297, 3.1415927],
             [-6.2831855, 6.2831855],
             [0, 0.044643],
             [0, 0.044643],]
        )  # joint limits in maniskill2
        self.gripper_limits = np.array([-10, 850]).astype(float)

        self.init_qpos = np.array(
            [0, 0, 0, np.pi / 3, 0, np.pi / 3, -np.pi / 2, 0.044643, 0.044643]
        )
        self.pose = Pose()  # base_pose
        self.safety_boundary = np.asarray(safety_boundary)
        self.boundary_clip_eps = boundary_clip_eps
        self.safety_boundary_clip = self.safety_boundary.copy()
        self.safety_boundary_clip[0::2] -= boundary_clip_eps
        self.safety_boundary_clip[1::2] += boundary_clip_eps
        self.with_hand_camera = with_hand_camera

        self.reset()

    def __del__(self):
        self.reset()
        self.arm.disconnect()

    def get_err_warn_code(self, show=False):
        code, (error_code, warn_code) = self.arm.get_err_warn_code(show=True)
        assert code == 0, "Failed to get_err_warn_code"
        return error_code, warn_code

    def clean_warning_error(self, mode=None):
        error_code, warn_code = self.get_err_warn_code(show=True)
        if warn_code != 0:
            self.arm.clean_warn()
        if error_code != 0:
            self.arm.clean_error()

        self.arm.motion_enable(enable=True)
        self.arm.set_mode(self.SUPPORTED_MOTION_MODES.index(self._motion_mode)
                          if mode is None else mode)
        self.arm.set_state(state=0)

    def reset(self, wait=True):
        self.clean_warning_error(mode=0)

        # NOTE: Remove satefy boundary during reset
        # boundary: [x_max, x_min, y_max, y_min, z_max, z_min]
        self.arm.set_reduced_tcp_boundary([999, -999, 999, -999, 999, -999])
        self.set_qpos(self.init_qpos, wait=wait)

        # boundary: [x_max, x_min, y_max, y_min, z_max, z_min]
        self.arm.set_reduced_tcp_boundary(self.safety_boundary)
        self.arm.set_tcp_load(0.82, [0.0, 0.0, 48.0])
        if self.with_hand_camera:
            self.arm.set_tcp_offset([0.0, 0.0, 177.0, 0.0, 0.0, 0.0])
        else:
            self.arm.set_tcp_offset([0.0, 0.0, 172.0, 0.0, 0.0, 0.0])
        self.arm.set_self_collision_detection(True)
        self.arm.set_collision_tool_model(1)  # xArm Gripper
        self.arm.set_collision_rebound(True)

        self.arm.motion_enable(enable=True)
        self.arm.set_mode(self.SUPPORTED_MOTION_MODES.index(self._motion_mode))
        self.arm.set_state(state=0)

    # def get_absolute_gripper_pos(self):
    #     _, gripper_pos = self.arm.get_gripper_position()
    #     return gripper_pos

    @property
    def robot(self):
        """An alias for compatibility."""
        return self

    @property
    def control_mode(self):
        """Get the currently activated controller uid."""
        return self._control_mode

    @property
    def motion_mode(self):
        """Get the currently activated controller uid."""
        return self._motion_mode

    @property
    def action_space(self):
        if self._control_mode == "pd_ee_pos":
            return spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        elif self._control_mode == "pd_ee_delta_pos":
            return spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        # elif self._control_mode == "pd_ee_pose":
        #     # [x, y, z, r, p, y, gripper], xyz in meters, rpy in radian
        #     return spaces.Box(low=np.array([-np.inf]*3 + [-np.pi]*3 + [-1]),
        #                       high=np.array([np.inf]*3 + [np.pi]*3 + [1]),
        #                       shape=(7,), dtype=np.float32)
        elif self._control_mode == "pd_ee_pose_axangle":
            # [x, y, z, *rotvec, gripper], xyz in meters
            #   rotvec is in axis of rotation and its norm gives rotation angle
            return spaces.Box(low=np.array([-np.inf]*6 + [-1]),
                              high=np.array([np.inf]*6 + [1]),
                              shape=(7,), dtype=np.float32)
        elif self._control_mode == "pd_ee_delta_pose_axangle":  # TODO: verify bounds
            # [x, y, z, *rotvec, gripper], xyz in meters
            #   rotvec is in axis of rotation and its norm gives rotation angle
            return spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        elif self._control_mode == "pd_ee_pose_quat":
            # [x, y, z, w, x, y, z, gripper], xyz in meters, wxyz is unit quaternion
            return spaces.Box(low=np.array([-np.inf]*3 + [-1] * 4 + [-1]),
                              high=np.array([np.inf]*3 + [1] * 4 + [1]),
                              shape=(8,), dtype=np.float32)
        elif self._control_mode == "pd_ee_delta_pose_quat":
            # [x, y, z, w, x, y, z, gripper], xyz in meters, wxyz is unit quaternion
            return spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        else:
            raise NotImplementedError(f"Unsupported {self._control_mode = }")

    # ---------------------------------------------------------------------- #
    # Control robot
    # ---------------------------------------------------------------------- #
    def _preprocess_action(self, action: np.ndarray,
                           translation_scale: float, axangle_scale: float):
        """Preprocess action:
            * apply translation_scale and axangle_scale
            * clip tgt_tcp_pose to avoid going out of safety_boundary
            * clip and rescale gripper action (action[-1])
        :param action: control action (unit for translation is meters)
                       If in delta control_mode, action needs to apply scale
        :return tgt_tcp_pose: target TCP pose in robot base frame (unit in mm)
        :return delta_tcp_pose: delta TCP pose in robot tool frame (unit in mm)
        :return gripper_pos: gripper action after rescaling [-10, 850] mm
        """
        cur_tcp_pose = self.get_tcp_pose(unit_in_mm=True)

        if self._control_mode == "pd_ee_pos":
            tgt_tcp_pose = Pose(p=action[:3] * 1000.0, q=cur_tcp_pose.q)  # m => mm
        elif self._control_mode == "pd_ee_delta_pos":
            delta_tcp_pos = action[:3] * translation_scale  # in milimeters
            tgt_tcp_pose = cur_tcp_pose * Pose(p=delta_tcp_pos)
        # elif self._control_mode == "pd_ee_pose":
        #     tgt_tcp_pose = Pose(action[:3] * 1000.0,
        #                         euler2quat(*action[3:6], axes='sxyz'))
        elif self._control_mode == "pd_ee_pose_axangle":
            axangle = action[3:6]
            rot_angle = np.linalg.norm(axangle)
            tgt_tcp_pose = Pose(p=action[:3] * 1000.0,  # m => mm
                                q=axangle2quat(axangle / (rot_angle + 1e-9),
                                               rot_angle))
        elif self._control_mode == "pd_ee_delta_pose_axangle":
            axangle = action[3:6]
            rot_angle = np.linalg.norm(axangle)
            delta_tcp_pose = Pose(p=action[:3] * translation_scale,  # in milimeters
                                  q=axangle2quat(axangle / (rot_angle + 1e-9),
                                                 rot_angle * axangle_scale))
            tgt_tcp_pose = cur_tcp_pose * delta_tcp_pose
        elif self._control_mode == "pd_ee_pose_quat":
            tgt_tcp_pose = Pose(p=action[:3] * 1000.0, q=action[3:7])  # m => mm
        elif self._control_mode == "pd_ee_delta_pose_quat":
            delta_tcp_pose = Pose(p=action[:3] * translation_scale,  # in milimeters
                                  q=action[3:7])  # in milimeters
            tgt_tcp_pose = cur_tcp_pose * delta_tcp_pose
        else:
            raise NotImplementedError()

        # Clip tgt_tcp_pose.p to safety_boundary_clip
        tgt_tcp_pose.set_p(np.clip(tgt_tcp_pose.p,
                                   self.safety_boundary_clip[1::2],
                                   self.safety_boundary_clip[0::2]))

        # [-1, 1] => [gripper_min, gripper_max]
        gripper_pos = clip_and_scale_action(action[-1], self.gripper_limits)

        delta_tcp_pose = cur_tcp_pose.inv() * tgt_tcp_pose

        self.logger.info(f"Setting {tgt_tcp_pose = }, {gripper_pos = }")
        return tgt_tcp_pose, delta_tcp_pose, gripper_pos

    def set_action(self, action: np.ndarray,
                   translation_scale=100.0, axangle_scale=0.1,
                   speed=None, mvacc=None,
                   skip_gripper=False, wait=False):
        """
        :param translation_scale: action [-1, 1] maps to [-100mm, 100mm]
        :param axangle_scale: axangle action norm (rotation angle) is multiplied by 0.1
                              [-1, 0, 0] => rotate around [1, 0, 0] by -0.1 rad
        :param speed: move speed
        :param mvacc: move acceleration
        :param skip_gripper: whether to skip gripper action
        """
        # TODO: Check if there's a way to not wait for set_gripper_position()
        #       so skip_gripper is not needed

        # Clean existing warnings / errors
        while self.arm.has_err_warn:
            error_code, warn_code = self.arm.get_err_warn_code()
            if error_code in [35]:  # 35: Safety Boundary Limit
                self.clean_warning_error()
            else:
                self.logger.error(f"ErrorCode: {error_code}, need to manually clean it")
                self.arm.get_err_warn_code(show=True)
                _ = input("Press enter after cleaning error")

        # Checks action shape and range
        assert action in self.action_space, f"Wrong {action = }"
        action = np.asarray(action).copy()  # TODO: can be removed?

        # Preprocess action (apply scaling, clip to safety boundary, rescale gripper)
        tgt_tcp_pose, delta_tcp_pose, gripper_pos = self._preprocess_action(
            action, translation_scale, axangle_scale
        )

        # Control gripper position
        ret_gripper = 0
        if not skip_gripper:
            ret_gripper = self.arm.set_gripper_position(gripper_pos, wait=wait)

        # Control xArm based on motion mode
        if self._motion_mode == "position":
            ret_arm = self.arm.set_tool_position(
                *delta_tcp_pose.p, *quat2euler(delta_tcp_pose.q, axes='sxyz'),
                is_radian=True, wait=wait
            )
            return ret_arm, ret_gripper
        elif self._motion_mode == "servo":
            raise NotImplementedError("Do not use servo mode! Need fine waypoints")
            ret_arm = self.arm.set_servo_cartesian(
                np.hstack([tgt_tcp_pose.p, quat2euler(tgt_tcp_pose.q, axes='sxyz')]),
                is_radian=True, is_tool_coord=False
            )
            return ret_arm, ret_gripper
        elif self._motion_mode == "joint_teaching":
            raise ValueError("Joint teaching mode enabled (no action needed)")
        elif self._motion_mode == "joint_vel":
            # clip delta pos to prevent undesired rotation due to ik solutions
            cur_tcp_pose = self.get_tcp_pose(unit_in_mm=True)
            tgt_tcp_pose.set_p(np.clip(tgt_tcp_pose.p,
                                       cur_tcp_pose.p - 30, cur_tcp_pose.p + 30))

            _, tgt_qpos = self.arm.get_inverse_kinematics(
                np.hstack([tgt_tcp_pose.p, quat2euler(tgt_tcp_pose.q, axes='sxyz')]),
                input_is_radian=True, return_is_radian=True
            )
            _, (cur_qpos, _, _) = self.arm.get_joint_states(is_radian=True)
            delta_qpos = np.asarray(tgt_qpos) - np.asarray(cur_qpos)

            timestep = 0.5
            qvel = delta_qpos / timestep
            qvel = np.clip(qvel, -0.3, 0.3)  # clip qvel for safety
            ret_arm = self.arm.vc_set_joint_velocity(
                qvel, is_radian=True, is_sync=True, duration=timestep
            )
            # time.sleep(timestep - 0.2)
            return ret_arm, ret_gripper
        elif self._motion_mode == "cartesian_vel":
            raise NotImplementedError(f"{self._motion_mode = } is not yet implemented")

            # [spd_x, spd_y, spd_z, spd_rx, spd_ry, spd_rz]
            speeds_xyz_rpy = np.zeros(6)
            ret_arm = self.arm.vc_set_cartesian_velocity(
                speeds_xyz_rpy, is_radian=True, is_tool_coord=True, duration=timestep
            )
            # time.sleep(timestep - 0.2)
            return ret_arm, ret_gripper
        elif self._motion_mode == "joint_online":
            _, tgt_qpos = self.arm.get_inverse_kinematics(
                np.hstack([tgt_tcp_pose.p, quat2euler(tgt_tcp_pose.q, axes='sxyz')]),
                input_is_radian=True, return_is_radian=True
            )
            ret_arm = self.arm.set_servo_angle(
                angle=tgt_qpos, speed=speed, mvacc=mvacc,
                relative=False, is_radian=True, wait=wait
            )
            return ret_arm, ret_gripper
        elif self._motion_mode == "cartesian_online":
            ret_arm = self.arm.set_position(
                *tgt_tcp_pose.p, *quat2euler(tgt_tcp_pose.q, axes='sxyz'),
                speed=speed, mvacc=mvacc,
                relative=False, is_radian=True, wait=wait
            )
            return ret_arm, ret_gripper
        else:
            raise NotImplementedError()

    def set_qpos(self, qpos, wait=False):
        """Set xarm qpos using maniskill2 qpos"""
        assert len(qpos) == 9, f"Wrong qpos shape: {len(qpos)}"
        arm_qpos, gripper_qpos = qpos[:7], qpos[-2:]
        ret_arm = self.arm.set_servo_angle(angle=arm_qpos, is_radian=True,
                                           wait=wait)

        gripper_qpos = gripper_qpos[0]  # NOTE: mimic action
        gripper_pos = clip_and_scale_action(
            gripper_qpos, self.gripper_limits, self.joint_limits_ms2[-1, :]
        )
        ret_gripper = self.arm.set_gripper_position(gripper_pos, wait=wait)
        return ret_arm, ret_gripper

    @staticmethod
    def build_grasp_pose(center, approaching=[0.0, 0.0, -1.0],
                         closing=[1.0, 0.0, 0.0]) -> Pose:
        center = np.asarray(center)
        approaching, closing = np.asarray(approaching), np.asarray(closing)
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return Pose.from_transformation_matrix(T)

    # ---------------------------------------------------------------------- #
    # Get robot information
    # ---------------------------------------------------------------------- #
    def get_qpos(self):
        """Get xarm qpos in maniskill2 format"""
        _, (qpos, qvel, effort) = self.arm.get_joint_states(is_radian=True)
        _, gripper_pos = self.arm.get_gripper_position()

        gripper_qpos = clip_and_scale_action(
            gripper_pos, self.joint_limits_ms2[-1, :], self.gripper_limits
        )
        return np.hstack([qpos, [gripper_qpos, gripper_qpos]])

    def get_qvel(self):
        """Get xarm qvel in maniskill2 format"""
        _, (qpos, qvel, effort) = self.arm.get_joint_states(is_radian=True)
        return np.hstack([qvel, [0.0, 0.0]])  # No gripper qvel

    def get_tcp_pose(self, unit_in_mm=False) -> Pose:
        """Get TCP pose in robot base frame
        :return pose: If unit_in_mm, position unit is mm. Else, unit is m.
        """
        _, (qpos, qvel, effort) = self.arm.get_joint_states(is_radian=True)
        _, base_to_tcp = self.arm.get_forward_kinematics(
            qpos, input_is_radian=True, return_is_radian=True
        )
        base_to_tcp = np.asarray(base_to_tcp)
        base_to_tcp_pose = Pose(
            p=base_to_tcp[:3] if unit_in_mm else base_to_tcp[:3] / 1000,
            q=euler2quat(*base_to_tcp[3:], axes='sxyz')
        )
        return base_to_tcp_pose

    # ---------------------------------------------------------------------- #
    # Observations
    # ---------------------------------------------------------------------- #
    def get_proprioception(self):
        obs = OrderedDict(qpos=self.get_qpos(), qvel=self.get_qvel())
        # controller_state = self.controller.get_state()
        # if len(controller_state) > 0:
        #     obs.update(controller=controller_state)
        return obs

    def get_state(self) -> dict:
        """Get current state, including robot state and controller state"""
        state = OrderedDict()

        # robot state
        state["robot_root_pose"] = vectorize_pose(self.pose)
        state["robot_qpos"] = self.get_qpos()
        state["robot_qvel"] = self.get_qvel()
        # state["robot_qacc"] = self.robot.get_qacc()

        # controller state
        # state["controller"] = self.controller.get_state()

        return state

    def set_state(self, state: dict, ignore_controller=False):
        # robot state
        pose_array = state["robot_root_pose"]
        self.pose = Pose(p=pose_array[:3], q=pose_array[3:])
        self.set_qpos(state["robot_qpos"])
        # self.robot.set_qvel(state["robot_qvel"])
        # self.robot.set_qacc(state["robot_qacc"])

        # if not ignore_controller and "controller" in state:
        #     self.controller.set_state(state["controller"])

    @property
    def cameras(self) -> CameraConfig:
        """CameraConfig of cameras attached to agent"""
        T_tcp_cam = Pose(p=[0, 0, 0.177]).inv() * Pose(
            p=[-0.06042734, 0.0175, 0.02915237],
            q=euler2quat(np.pi, -np.pi/2-np.pi/12, np.pi)
        ) * Pose(p=[0, 0.015, 0])  # camera_color_frame
        return CameraConfig(
            uid="hand_camera",
            device_sn="146322076186",
            pose=T_tcp_cam,
            width=848,
            height=480,
            preset="High Accuracy",
            depth_option_kwargs={rs.option.exposure: 1500},
            actor_pose_fn=self.get_tcp_pose,
        )