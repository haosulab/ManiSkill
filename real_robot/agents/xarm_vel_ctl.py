import numpy as np
from sapien.core import Pose
from xarm.wrapper import XArmAPI
from real_robot.utils.logger import get_logger
from transforms3d.quaternions import axangle2quat
from transforms3d.euler import euler2quat, quat2euler
import time


class XArm7:
    def __init__(self, ip="192.168.1.229", control_mode='pd_ee_delta_pos', servo=False, 
                 velocity_ctrl=False, cartesian_velocity_ctrl=False, 
                 ignore_gripper_action=False, ms2_transform=False,
                 use_ms2_ik=False, debug=False):
        self.logger = get_logger("XArm7")
        self.arm = XArmAPI(ip, is_radian=True)
        self.servo = servo
        assert not self.servo, "Dangerous"
        self.servo_args = {'speed': 100, 'mvacc': 2000}
        self.velocity_ctrl = velocity_ctrl
        self.cartesian_velocity_ctrl = cartesian_velocity_ctrl
        # 0: position control; 1: servo control; 5: cartesian velocity control
        if self.servo:
            self.arm_mode = 1
        elif self.velocity_ctrl:
            self.arm_mode = 4
        elif self.cartesian_velocity_ctrl:
            self.arm_mode = 5
        else:
            self.arm_mode = 0
        self.ignore_gripper_action = ignore_gripper_action
        self.debug = debug

        self.ms2_transform = ms2_transform # whether to translate from/to maniskill2 qpos / action representations
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
        self.gripper_ms2_action_qpos_mapping = np.array([0.0, 0.0446430]) # ms2 action -1 maps to ms2 gripper qpos -0.01; ms2 action 1 maps to ms2 gripper qpos 0.044
        if self.ms2_transform:
            self.init_qpos = np.array(
                [0, 0, 0, np.pi / 3, 0, np.pi / 3, -np.pi / 2, 0.044643, 0.044643]
            )
        else:
            self.init_qpos = np.array(
                [0, 0, 0, np.pi / 3, 0, np.pi / 3, -np.pi / 2, self.gripper_limits[1], self.gripper_limits[1]]
            )

        self._control_mode = control_mode
        assert self._control_mode in ["pd_ee_delta_pos", "pd_ee_delta_pose_axangle", "pd_ee_delta_pose_quat",
                                      "pd_ee_pos", "pd_ee_pose_axangle", "pd_ee_pose_quat"]

        # Avoid hitting the table or moving robot arm too far that it causes kinematic errors
        self.tgt_tcp_bounds_min = np.array([0.20, -0.40, 0.01]) # xmin, ymin, zmin
        self.tgt_tcp_bounds_max = np.array([0.55, 0.05, 0.28]) # xmax, ymax, zmax
        
        # Whether to use maniskill2 to compute inverse kinematics; if not, XArm's inverse kinematics is used
        self.use_ms2_ik = use_ms2_ik
        if self.use_ms2_ik:
            self.logger.info("Using ManiSkill2 IK")
            import mani_skill2.envs, gym
            self.ms2_dummy_env = gym.make('PickCube-v0', robot='xarm7', control_mode='pd_ee_delta_pose')
            self.ms2_dummy_env.reset()
            self.ms2_ik_fxn = self.ms2_dummy_env.agent.controller.controllers['arm'].compute_ik
        else:
            self.ms2_dummy_env = None
            self.ms2_ik_fxn = None

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
        self.arm.set_state(state=0)
        if mode is None:
            self.arm.set_mode(self.arm_mode)
        else:
            self.arm.set_mode(mode)
        self.arm.set_state(state=0)
        time.sleep(0.5) # wait for set_mode to be done

    def reset(self, wait=True):
        self.clean_warning_error(mode=0)

        # NOTE: Remove satefy boundary during reset
        # boundary: [x_max, x_min, y_max, y_min, z_max, z_min]
        self.arm.set_reduced_tcp_boundary([999, -999, 999, -999, 999, -999])
        self.set_qpos(self.init_qpos, wait=wait)

        # boundary: [x_max, x_min, y_max, y_min, z_max, z_min]
        self.arm.set_reduced_tcp_boundary([999, -999, 999, -999, 999, 0])
        self.arm.set_tcp_load(0.82, [0.0, 0.0, 48.0])
        self.arm.set_tcp_offset([0.0, 0.0, 172.0, 0.0, 0.0, 0.0])
        self.arm.set_self_collision_detection(True)
        self.arm.set_collision_tool_model(1)  # xArm Gripper
        self.arm.set_collision_rebound(True)

        self.arm.motion_enable(enable=True)
        self.arm.set_state(state=0)
        self.arm.set_mode(self.arm_mode)
        self.arm.set_state(state=0) # set state after set mode, otherwise code=9 error will occur
        time.sleep(0.5) # wait for set_mode to be done

    def xarm_gripper_qpos_from_ms2_gripper_qpos(self, gripper_qpos):
        xarm_gripper_qpos = (
            (gripper_qpos - self.joint_limits_ms2[-1, 0])
            / (self.joint_limits_ms2[-1, 1] - self.joint_limits_ms2[-1, 0])
            * (self.gripper_limits[1] - self.gripper_limits[0])
            + self.gripper_limits[0]
        )
        return xarm_gripper_qpos

    def ms2_gripper_qpos_from_xarm_gripper_qpos(self, gripper_qpos):
        ms2_gripper_qpos = (
            (gripper_qpos - self.gripper_limits[0])
            / (self.gripper_limits[1] - self.gripper_limits[0])
            * (self.joint_limits_ms2[-1, 1] - np.maximum(self.joint_limits_ms2[-1, 0], 0))
            + np.maximum(self.joint_limits_ms2[-1, 0], 0)
        ) # ManiSkill2 qpos is non-negative
        return ms2_gripper_qpos

    def xarm_gripper_action_from_ms2_gripper_action(self, gripper_action): # -1: close; 1: open
        ms2_gripper_target_qpos = (
            (gripper_action + 1) / 2
            * (self.gripper_ms2_action_qpos_mapping[1] - self.gripper_ms2_action_qpos_mapping[0])
            + self.gripper_ms2_action_qpos_mapping[0]
        )
        ms2_gripper_target_qpos = np.clip(ms2_gripper_target_qpos, self.joint_limits_ms2[-1, 0], self.joint_limits_ms2[-1, 1])
        xarm_gripper_action = self.xarm_gripper_qpos_from_ms2_gripper_qpos(ms2_gripper_target_qpos)
        return xarm_gripper_action

    def get_absolute_gripper_pos(self):
        _, gripper_pos = self.arm.get_gripper_position()
        return gripper_pos

    def set_action(self, action, translation_scale=100.0, axangle_scale=0.1, wait=False):
        """
        :param translation_scale: e.g. 100.0 =>  [-1, 1] maps to [-100mm, 100mm]
        :param axangle_scale: e.g., 0.1 => angle encoded by the rotation component of action is multiplied by 0.1 (axis remains the same)
        """
        while self.arm.has_err_warn:
            error_code, warn_code = self.arm.get_err_warn_code()
            if error_code in [35]:
                self.clean_warning_error()
            else:
                self.logger.error(f"ErrorCode: {error_code}, need to manually clean it")
                self.arm.get_err_warn_code(show=True)
                _ = input("Press enter after cleaning error")

        if self._control_mode in ["pd_ee_delta_pos", "pd_ee_pos"]:
            assert action.size == 4
            action = np.concatenate([action[:3], [1, 0, 0, 0], action[-1:]])
        elif self._control_mode in ['pd_ee_delta_pose_axangle', 'pd_ee_pose_axangle']:
            assert action.size == 7
            axangle = action[3:6]
            ax = axangle / (np.linalg.norm(axangle) + 1e-6)
            angle = np.linalg.norm(axangle) * axangle_scale
            action = np.concatenate([action[:3], axangle2quat(ax, angle), action[-1:]])
        elif self._control_mode in ['pd_ee_delta_pose_quat', 'pd_ee_pose_quat']:
            assert action.size == 8
        else:
            raise NotImplementedError()

        cur_tcp_pose = self.get_tcp_pose()
        cur_tcp_pose = Pose(p=cur_tcp_pose[:3], q=cur_tcp_pose[3:])
        
        if 'delta' in self._control_mode:
            delta_xyz = action[:3] * translation_scale # in milimeters
            delta_quat = action[3:7]
            tgt_tcp_pose = cur_tcp_pose * Pose(p=delta_xyz / 1000, q=delta_quat) # p in meters
            tgt_tcp_pose_p, tgt_tcp_pose_q = tgt_tcp_pose.p, tgt_tcp_pose.q
            tgt_tcp_pose_p = np.clip(tgt_tcp_pose_p, self.tgt_tcp_bounds_min, self.tgt_tcp_bounds_max)
        else:
            tgt_tcp_pose_p = action[:3] * translation_scale / 1000 # in meters
            tgt_tcp_pose_p = np.clip(tgt_tcp_pose_p, self.tgt_tcp_bounds_min, self.tgt_tcp_bounds_max)
            tgt_tcp_pose_q = action[3:7]
            
        actual_delta_pose = cur_tcp_pose.inv() * Pose(p=tgt_tcp_pose_p, q=tgt_tcp_pose_q)
        actual_delta_xyz = actual_delta_pose.p * 1000 # in milimeters
        actual_delta_quat = actual_delta_pose.q
            
        if self.ms2_transform:
            # [-1, 1] => [gripper_max, gripper_min]
            gripper_action = self.xarm_gripper_action_from_ms2_gripper_action(gripper_action=action[-1])
        else:
            gripper_action = action[-1]
        self.logger.info(f"{tgt_tcp_pose_p = } {tgt_tcp_pose_q = } {actual_delta_quat = } {gripper_action = }")

        ret_gripper = -1
        if not self.ignore_gripper_action:
            ret_gripper = self.arm.set_gripper_position(gripper_action, wait=wait) # this waits execution regardless of wait=True/False...

        if self.servo:
            assert self._control_mode in ["pd_ee_delta_pos", "pd_ee_pos"]
            actual_delta_xyz_norm = np.linalg.norm(actual_delta_xyz)
            actual_delta_xyz = actual_delta_xyz / np.clip(actual_delta_xyz_norm, 1e-3, None) * np.clip(actual_delta_xyz_norm, 1e-3, 5.0)
            ret_arm = self.arm.set_servo_cartesian(
                np.concatenate([actual_delta_xyz, np.zeros(4)]),
                is_tool_coord=True,
                **self.servo_args
            )
            time.sleep(0.2)
        elif self.velocity_ctrl:
            if not self.use_ms2_ik:
                tgt_tcp_pose_p = np.clip(tgt_tcp_pose_p, cur_tcp_pose.p - 0.03, cur_tcp_pose.p + 0.03) # clip delta pos to prevent undesired rotation due to ik solutions
                _, tgt_joint_states = self.arm.get_inverse_kinematics(
                    np.concatenate([tgt_tcp_pose_p * 1000, quat2euler(tgt_tcp_pose_q, axes='sxyz')]), 
                    input_is_radian=True, return_is_radian=True)
            else:
                self.ms2_dummy_env.set_qpos(np.concatenate([self.get_qpos()[:-2], 0.0, 0.0]))
                tgt_tcp_pose_p = np.clip(tgt_tcp_pose_p, cur_tcp_pose.p - 0.10, cur_tcp_pose.p + 0.10)
                tgt_joint_states = self.ms2_ik_fxn(Pose(p=tgt_tcp_pose_p, q=tgt_tcp_pose_q))
            _, (cur_joint_states, _, _) = self.arm.get_joint_states(is_radian=True)
            diff_joint_states = np.array(tgt_joint_states) - np.array(cur_joint_states)
            timestep = 0.5
            joint_vel = diff_joint_states / timestep
            joint_vel_clipped = np.clip(joint_vel, -0.3, 0.3)
            ret_arm = self.arm.vc_set_joint_velocity(joint_vel_clipped, is_radian=True, is_sync=True, duration=timestep)
            # time.sleep(timestep - 0.2)
        elif self.cartesian_velocity_ctrl:
            assert self._control_mode in ["pd_ee_delta_pos", "pd_ee_pos"]
            actual_delta_xyz_norm = np.linalg.norm(actual_delta_xyz)
            actual_delta_xyz = actual_delta_xyz / np.clip(actual_delta_xyz_norm, 1e-3, None) * np.clip(actual_delta_xyz_norm, 1e-3, 50.0)
            ret_arm = self.arm.vc_set_cartesian_velocity(
                np.concatenate([actual_delta_xyz, np.zeros(3)]), 
                is_tool_coord=True, 
                duration=1.0,
            )
            time.sleep(0.2)
        else:
            assert self._control_mode in ["pd_ee_delta_pos", "pd_ee_pos"]
            ret_arm = self.arm.set_tool_position(
                x=actual_delta_xyz[0], y=actual_delta_xyz[1], z=actual_delta_xyz[2],
                wait=wait
            ) # if wait=False, this adds action to the queue, and action will be exec to the full following orders in the queue
        return ret_arm, ret_gripper

    def set_qpos(self, qpos, wait=False):
        """Set xarm qpos using maniskill2 qpos"""
        arm_qpos, gripper_qpos = qpos[:7], qpos[-2:]
        ret_arm = self.arm.set_servo_angle(angle=arm_qpos, is_radian=True,
                                           wait=wait)

        gripper_qpos = gripper_qpos[0]  # NOTE: mimic action
        if self.ms2_transform:
            gripper_pos = self.xarm_gripper_qpos_from_ms2_gripper_qpos(gripper_qpos)
        else:
            gripper_pos = gripper_qpos
        # print("Gripper pos:", gripper_pos)

        ret_gripper = -1
        if not self.ignore_gripper_action:
            ret_gripper = self.arm.set_gripper_position(gripper_pos, wait=wait)
        return ret_arm, ret_gripper

    def get_qpos(self):
        """Get xarm qpos in maniskill2 format"""
        _, (qpos, qvel, effort) = self.arm.get_joint_states(is_radian=True)
        _, gripper_pos = self.arm.get_gripper_position()

        if self.ms2_transform:
            gripper_qpos = self.ms2_gripper_qpos_from_xarm_gripper_qpos(gripper_pos)
        return np.hstack([qpos, [gripper_qpos, gripper_qpos]])

    def get_qvel(self):
        """Get xarm qvel in maniskill2 format"""
        _, (qpos, qvel, effort) = self.arm.get_joint_states(is_radian=True)
        return np.hstack([qvel, [0.0, 0.0]])  # No gripper qvel

    def get_base_to_tcp_6d(self):
        _, (qpos, qvel, effort) = self.arm.get_joint_states(is_radian=True)
        _, base_to_tcp = self.arm.get_forward_kinematics(
            qpos, input_is_radian=True, return_is_radian=True
        )
        base_to_tcp = np.array(base_to_tcp) # [6,], pos in meters and rot in radians
        return base_to_tcp

    def get_tcp_pose(self):
        base_to_tcp = self.get_base_to_tcp_6d()
        base_to_tcp[:3] = base_to_tcp[:3] / 1000  # mm to m
        # print("base_to_joint7", base_to_joint7)
        base_to_tcp_pose = Pose(
            p=base_to_tcp[:3],
            q=euler2quat(*base_to_tcp[3:6], axes='sxyz')
        )
        return np.hstack([base_to_tcp_pose.p, base_to_tcp_pose.q])
