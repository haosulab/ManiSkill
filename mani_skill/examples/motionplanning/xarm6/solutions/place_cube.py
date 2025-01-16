import numpy as np
import sapien

from mani_skill.envs.tasks import PlaceCubeEnv
from mani_skill.examples.motionplanning.xarm6.motionplanner import \
    XArm6MotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env: PlaceCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = XArm6MotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    invisible_offset = 0.068
    env = env.unwrapped

    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(env.obj)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.obj.pose.sp.p)
    # grasp_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, invisible_offset]), grasp_pose.q)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Move above bin
    # -------------------------------------------------------------------------- #
    position_above_bin = env.bin.pose.sp.p + np.array([0, 0, 2 * (env.cube_half_length + env.block_half_size[0]) + 8*env.short_side_half_size])
    above_bin_pose = sapien.Pose(position_above_bin, grasp_pose.q)
    planner.move_to_pose_with_screw(above_bin_pose)

    # -------------------------------------------------------------------------- #
    # Move down
    # -------------------------------------------------------------------------- #
    position_inside_bin = env.bin.pose.sp.p + np.array([0, 0, 2 * (env.cube_half_length + env.block_half_size[0]) + 2*env.short_side_half_size])
    inside_bin_pose = sapien.Pose(position_inside_bin, grasp_pose.q)
    planner.move_to_pose_with_screw(inside_bin_pose)

    # -------------------------------------------------------------------------- #
    # Open gripper
    # -------------------------------------------------------------------------- #
    res = planner.open_gripper()

    planner.close()
    return res

