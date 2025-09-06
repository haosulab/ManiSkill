import numpy as np
import sapien
import mplib
from mani_skill.examples.motionplanning.two_finger_gripper.motionplanner import TwoFingerGripperMotionPlanningSolver
from mani_skill.envs.sapien_env import BaseEnv

class XArm6RobotiqMotionPlanningSolver(TwoFingerGripperMotionPlanningSolver):
    CLOSED = 0.81
    OPEN = 0

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
    ):
        super().__init__(env, debug, vis, base_pose, visualize_target_grasp_pose, print_env_info, joint_vel_limits, joint_acc_limits)