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
    FINGER_LENGTH = 0.025
    env = env.unwrapped

    moving_cube = env.cubeA
    target_cube = env.cubeB


    # -------------------------------------------------------------------------- #
    # Move the specified cube to be next to the other cube
    # -------------------------------------------------------------------------- #
    # Move Gripper to the specified cube
    obb = get_actor_obb(moving_cube)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    distance = np.linalg.norm(moving_cube.pose.sp.p - target_cube.pose.sp.p)  

    need_move_a_b = (distance > 0.07)
    if need_move_a_b:
        planner.close_gripper()
        grasp_pose = env.agent.build_grasp_pose(approaching, closing, moving_cube.pose.sp.p)

        # Reach
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        planner.move_to_pose_with_screw(reach_pose)

        # Grasp
        planner.move_to_pose_with_screw(grasp_pose)
        planner.close_gripper()

        # Move to Goal Pose
        goal_pose = sapien.Pose(target_cube.pose.sp.p * 0.8, grasp_pose.q)
        planner.move_to_pose_with_screw(goal_pose)

    # -------------------------------------------------------------------------- #
    # Stack Cube C onto Cube A and B
    # -------------------------------------------------------------------------- #

    obb = get_actor_obb(env.cubeC)
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # Search a valid pose
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Fail to find a valid grasp pose")

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #

    # planner.planner.update_attached_box([0.04, 0.04, 0.04], Pose.create(env.cubeB.pose).raw_pose.numpy().astype(np.float64).reshape(7,1))

    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)
    if need_move_a_b:
         planner.open_gripper()

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)

    # -------------------------------------------------------------------------- #
    # Stack
    # -------------------------------------------------------------------------- #
    goal_pose_A = env.cubeA.pose * sapien.Pose([0, 0, env.cube_half_size[2] * 2])
    goal_pose_B = env.cubeB.pose * sapien.Pose([0, 0, env.cube_half_size[2] * 2])
    goal_pose_p = (goal_pose_A.p + goal_pose_B.p)/2
    offset = (goal_pose_p - env.cubeC.pose.p).cpu().numpy()[0] # remember that all data in ManiSkill is batched and a torch tensor
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)

    res = planner.open_gripper()
    planner.close()
    return res

