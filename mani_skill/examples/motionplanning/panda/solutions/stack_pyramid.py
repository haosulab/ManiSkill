import argparse
import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import StackPyramidEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils.structs import Pose
from mani_skill.utils.logging_utils import logger

FINGER_LENGTH = 0.025

def move_and_grasp(cube, env, planner):
    """Moves to the cube, checks for collisions, and attempts a grasp."""
    obb = get_actor_obb(cube)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    
    # Try alternative angles if collision occurs
    angles = np.linspace(0, np.pi, 4)
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        adjusted_grasp_pose = grasp_pose * delta_pose
        res = planner.move_to_pose_with_screw(adjusted_grasp_pose, dry_run=True)
        if res == -1:
            continue

        # Reach to grasp pose
        planner.move_to_pose_with_screw(adjusted_grasp_pose)
        
        # Attempt to grasp
        planner.close_gripper()

        # Check if the object is grasped
        if is_object_grasped(cube, env, planner):
            logger.debug(f"Successfully grasped {cube}")
            return True, adjusted_grasp_pose  # Successful grasp
        else:
            # Open gripper if grasp failed and try the next angle
            planner.open_gripper()
    logger.debug(f"Failed to grasp {cube}")
    return False, None

def solve(env: StackPyramidEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    env = env.unwrapped

    # -------------------------------------------------------------------------- #
    # Push Cube A to be next to Cube B
    # -------------------------------------------------------------------------- #
    distance = np.linalg.norm(env.cubeA.pose.sp.p, axis=0) - np.linalg.norm(env.cubeB.pose.sp.p, axis=0)
    threshold = 0.025
    if (distance >= threshold):
        logger.debug(f"Distance >= {threshold}: {distance}")
        success, grasp_pose_A = move_and_grasp(env.cubeA, env, planner)
        if success:
            goal_pose = sapien.Pose(env.cubeB.pose.sp.p, grasp_pose_A.q)
            planner.move_to_pose_with_screw(goal_pose)
            planner.open_gripper()

    # -------------------------------------------------------------------------- #
    # Stack Cube C onto Cube A and B
    # -------------------------------------------------------------------------- #
    success, grasp_pose_C = move_and_grasp(env.cubeC, env, planner)
    if success:
        goal_pose_A = env.cubeA.pose * sapien.Pose([0, 0, env.cube_half_size[2] * 2])
        goal_pose_B = env.cubeB.pose * sapien.Pose([0, 0, env.cube_half_size[2] * 2])
        goal_pose_p = (goal_pose_A.p + goal_pose_B.p) / 2
        offset = (goal_pose_p - env.cubeC.pose.p).numpy()[0]  # remember that all data in ManiSkill is batched and a torch tensor
        align_pose = sapien.Pose(grasp_pose_C.p + offset, grasp_pose_C.q)
        planner.move_to_pose_with_screw(align_pose)

        res = planner.open_gripper()
        planner.close()
        return res
    else:
        logger.debug("Failed to grasp Cube C")
        planner.close()
        return -1
