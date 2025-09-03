import mplib
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.base_motionplanner.motionplanner import BaseMotionPlanningSolver


class PandaStickMotionPlanningSolver(BaseMotionPlanningSolver):
    MOVE_GROUP_LINKS = 7

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
    
    def setup_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand_tcp",
            joint_vel_limits=np.ones(self.MOVE_GROUP_LINKS) * self.joint_vel_limits,
            joint_acc_limits=np.ones(self.MOVE_GROUP_LINKS) * self.joint_acc_limits,
        )
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        return planner
        