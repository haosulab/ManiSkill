import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import PickCubeEnv
from mani_skill.examples.motionplanning.so100.motionplanner import \
    SO100ArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env: PickCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = SO100ArmMotionPlanningSolver(
        env,
        debug=False,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(env.cube)

    approaching = np.array([0, 0, -1])

    # rotate around x-axis to align with the expected frame for computing grasp poses (Z is up/down)
    tcp_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0)) * env.agent.tcp_pose.sp
    target_closing = (tcp_pose ).to_transformation_matrix()[:3, 1]
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.cube.pose.sp.p)

    # due to how SO100 is defined we need to transform the grasp pose back to what is expected by SO100
    grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(-np.pi/2, 0*np.pi / 2, np.pi / 2))

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    planner.gripper_state = 0
    reach_pose = sapien.Pose([0, 0, 0.03]) * grasp_pose
    planner.move_to_pose_with_screw(reach_pose)
    
    # reach_pose = sapien.Pose([0, 0, -0.02]) * env.agent.tcp_pose.sp
    # planner.move_to_pose_with_screw(sapien.Pose(reach_pose.p, env.agent.tcp_pose.sp.q))

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(sapien.Pose([0, 0, 0.01]) * grasp_pose)
    planner.close_gripper(gripper_state=-0.8)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(env.goal_site.pose.sp.p, grasp_pose.q)
    res = planner.move_to_pose_with_screw(goal_pose)
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res
