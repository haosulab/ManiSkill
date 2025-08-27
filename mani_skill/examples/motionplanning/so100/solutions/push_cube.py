import numpy as np
import sapien

from mani_skill.envs.tasks import PushCubeEnv
from mani_skill.examples.motionplanning.so100.motionplanner import \
    SO100ArmMotionPlanningSolver

def solve(env: PushCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = SO100ArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.01
    env = env.unwrapped
    planner.close_gripper()
    reach_pose = sapien.Pose(p=env.obj.pose.sp.p + np.array([-0.05, 0, 0]), q=env.agent.tcp_pose.sp.q)
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(p=env.goal_region.pose.sp.p + np.array([-0.12, 0, 0]),q=env.agent.tcp_pose.sp.q)
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res
