from collections import OrderedDict

import numpy as np
from gym import spaces
import pyrealsense2 as rs
from sapien.core import Pose
from transforms3d.euler import euler2quat, quat2euler

from xarm.wrapper import XArmAPI
from real_robot.utils.logger import get_logger
from real_robot.utils.common import clip_and_scale_action, vectorize_pose
from real_robot.sensors.camera import CameraConfig


class XArm7:
    """
    XArm7 agent class
    Mimics mani_skill2.agents.base_agent.BaseAgent interface
    """
    SUPPORTED_CONTROL_MODES = ("pd_ee_delta_pos", "pd_ee_pose")

    def __init__(self,
                 ip="192.168.1.229",
                 control_mode="pd_ee_delta_pos",
                 safety_boundary=[999, -999, 999, -999, 999, 0],
                 boundary_clip_eps=10):
        """
        :param safety_boundary: [x_max, x_min, y_max, y_min, z_max, z_min] (mm)
        :param boundary_clip_eps: clip action when TCP position to boundary is
                                  within boundary_clip_eps (mm)
        """
        self.logger = get_logger("XArm7")
        self.arm = XArmAPI(ip)

        if control_mode not in self.SUPPORTED_CONTROL_MODES:
            raise NotImplementedError(f"Unsupported {control_mode = }")
        self._control_mode = control_mode

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

        self.reset()

    def __del__(self):
        self.reset()
        self.arm.disconnect()

    def get_err_warn_code(self, show=False):
        code, (error_code, warn_code) = self.arm.get_err_warn_code(show=True)
        assert code == 0, "Failed to get_err_warn_code"
        return error_code, warn_code

    def clean_warning_error(self):
        error_code, warn_code = self.get_err_warn_code(show=True)
        if warn_code != 0:
            self.arm.clean_warn()
        if error_code != 0:
            self.arm.clean_error()

        self.arm.motion_enable(enable=True)
        self.arm.set_state(state=0)
        self.arm.set_mode(0)  # position control mode

    def reset(self, wait=True):
        self.clean_warning_error()

        # NOTE: Remove satefy boundary during reset
        # boundary: [x_max, x_min, y_max, y_min, z_max, z_min]
        self.arm.set_reduced_tcp_boundary([999, -999, 999, -999, 999, -999])
        self.set_qpos(self.init_qpos, wait=wait)

        # boundary: [x_max, x_min, y_max, y_min, z_max, z_min]
        self.arm.set_reduced_tcp_boundary(self.safety_boundary)
        self.arm.set_tcp_load(0.82, [0.0, 0.0, 48.0])
        self.arm.set_tcp_offset([0.0, 0.0, 172.0, 0.0, 0.0, 0.0])
        self.arm.set_self_collision_detection(True)
        self.arm.set_collision_tool_model(1)  # xArm Gripper
        self.arm.set_collision_rebound(True)

        self.arm.motion_enable(enable=True)
        self.arm.set_state(state=0)
        self.arm.set_mode(0)  # position control mode

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
    def action_space(self):
        if self._control_mode == "pd_ee_delta_pos":
            return spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        elif self._control_mode == "pd_ee_pose":
            # [x, y, z, r, p, y, gripper], xyz in meters, rpy in radian
            return spaces.Box(low=np.array([-np.inf]*3 + [-np.pi]*3 + [-1]),
                              high=np.array([np.inf]*3 + [np.pi]*3 + [1]),
                              shape=(7,), dtype=np.float32)
        else:
            raise NotImplementedError(f"Unsupported {self._control_mode = }")

    # ---------------------------------------------------------------------- #
    # Control robot
    # ---------------------------------------------------------------------- #
    def _preprocess_action(self, action: np.ndarray, action_scale: float):
        """Preprocess action to avoid going out of safety_boundary"""
        # TODO:combine action_scale with gripper_limits to do mapping together
        if self._control_mode == "pd_ee_delta_pos":
            delta_tcp_pos = action[:3] * action_scale  # in milimeters
            cur_tcp_pose = self.get_tcp_pose(unit_in_mm=True)
            tgt_tcp_pose = cur_tcp_pose * Pose(p=delta_tcp_pos)

            # Clip tgt_tcp_pose.p to safety_boundary_clip
            tgt_tcp_pose.set_p(
                np.clip(tgt_tcp_pose.p,
                        self.safety_boundary_clip[1::2],
                        self.safety_boundary_clip[0::2])
            )

            action[:3] = (cur_tcp_pose.inv() * tgt_tcp_pose).p
            # [-1, 1] => [gripper_max, gripper_min]
            action[-1] = clip_and_scale_action(action[-1], self.gripper_limits)
        elif self._control_mode == "pd_ee_pose":
            cur_tcp_pose = self.get_tcp_pose(unit_in_mm=True)
            tgt_tcp_pose = Pose(action[:3] * action_scale,
                                euler2quat(*action[3:6], axes='sxyz'))
            # Clip tgt_tcp_pose.p to safety_boundary_clip
            tgt_tcp_pose.set_p(
                np.clip(tgt_tcp_pose.p,
                        self.safety_boundary_clip[1::2],
                        self.safety_boundary_clip[0::2])
            )

            delta_tcp_pose = cur_tcp_pose.inv() * tgt_tcp_pose
            action[:6] = np.hstack([delta_tcp_pose.p,
                                    quat2euler(delta_tcp_pose.q, axes='sxyz')])
            # [-1, 1] => [gripper_max, gripper_min]
            action[-1] = clip_and_scale_action(action[-1], self.gripper_limits)
        else:
            raise NotImplementedError()
        self.logger.info(f"Setting {action = }")
        return action

    def set_action(self, action, action_scale=100.0, wait=False):
        """
        :param action_scale: action [-1, 1] maps to [-100mm, 100mm]
        """
        while self.arm.has_err_warn:
            error_code, warn_code = self.arm.get_err_warn_code()
            if error_code in [35]:  # 35: Safety Boundary Limit
                self.clean_warning_error()
            else:
                self.logger.error(f"ErrorCode: {error_code}, "
                                  "need to manually clean it")
                self.arm.get_err_warn_code(show=True)
                _ = input("Press enter after cleaning error")

        # Checks action shape and range
        assert action in self.action_space, f"Wrong {action = }"
        action = action.copy()

        if self._control_mode == "pd_ee_delta_pos":
            action = self._preprocess_action(action, action_scale)
            arm_delta_pos, gripper_pos = action[:3], action[-1]

            ret_gripper = self.arm.set_gripper_position(gripper_pos, wait=wait)
            ret_arm = self.arm.set_tool_position(*arm_delta_pos, wait=wait)
            return ret_arm, ret_gripper
        elif self._control_mode == "pd_ee_pose":
            action = self._preprocess_action(action, 1000.0)  # m => mm
            arm_delta_pose, gripper_pos = action[:6], action[-1]

            ret_gripper = self.arm.set_gripper_position(gripper_pos, wait=wait)
            ret_arm = self.arm.set_tool_position(*arm_delta_pose,
                                                 is_radian=True, wait=wait)
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
        center = np.array(center)
        approaching, closing = np.array(approaching), np.array(closing)
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
        T_tcp_cam = Pose(p=[0, 0, 0.172]).inv() * Pose(
            p=[-0.06042734, 0.0175, 0.02915237],
            q=euler2quat(np.pi, -np.pi/2-np.pi/12, np.pi)
        )
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
