import mplib
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.pose import to_sapien_pose
from transforms3d import euler
from mani_skill.examples.motionplanning.two_finger_gripper.motionplanner import TwoFingerGripperMotionPlanningSolver


class SO100ArmMotionPlanningSolver (TwoFingerGripperMotionPlanningSolver):
    OPEN = 0
    CLOSED = -0.8
    MOVE_GROUP = "Fixed_Jaw_tip"

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
    ):
        super().__init__(env, debug, vis, base_pose, visualize_target_grasp_pose, print_env_info, joint_vel_limits, joint_acc_limits)
        self._so_100_visual_grasp_pose_transform = sapien.Pose(q=euler.euler2quat(0, -np.pi / 2, np.pi / 2))

    @property
    def _so_100_grasp_pose_tcp_transform(self):
        return self.base_env.agent.robot.links_map["Fixed_Jaw_tip"].pose.sp * self.base_env.agent.tcp_pose.sp.inv()

    def _update_grasp_visual(self, target: sapien.Pose) -> None:
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(target * self._so_100_visual_grasp_pose_transform)

    def _transform_pose_for_planning(self, target: sapien.Pose) -> sapien.Pose:
        return sapien.Pose(p=target.p + self._so_100_grasp_pose_tcp_transform.p, q=target.q)

