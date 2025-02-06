import numpy as np
import sapien

from mani_skill.envs.tasks import PickAndPlaceEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb

def solve(env: PickAndPlaceEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    base_half_size = 0.25
    # height = 0.05
    height = 0.045
    planner.add_box_collision(extents=np.array([base_half_size, base_half_size, height]), pose=env.container_grid.pose.sp)

    # FINGER_LENGTH = 0.025
    FINGER_LENGTH = 0.03
    
    env = env.unwrapped
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Iterate over each cube and move it to the goal site

    order_of_cubes = list(range(len(env.cubes)))
    
    rng.shuffle(order_of_cubes)        
    cube_done = 0
    for i in order_of_cubes:
        cube = env.cubes[i]
        goal_site = env.goal_sites[i]
        # Retrieve the object oriented bounding box (trimesh box object)
        obb = get_actor_obb(cube)

        approaching = np.array([0, 0, -1])
        # Get transformation matrix of the tcp pose, is default batched and on torch
        target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        # Build a simple grasp pose using this information for Panda
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=FINGER_LENGTH,
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        
        grasp_pose = env.agent.build_grasp_pose(approaching, closing, cube.pose.sp.p)

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
        # Move to goal pose
        # -------------------------------------------------------------------------- #
        goal_pose_offset = np.array([0.0, 0.0, 0.14])
        goal_pose = sapien.Pose(goal_site.pose.sp.p + goal_pose_offset, grasp_pose.q)
        planner.move_to_pose_with_screw(goal_pose)
        cube_done += 1

        if (cube_done == (len(env.cubes))):
            res = planner.open_gripper()
        else:
            planner.open_gripper()
    
    planner.close()
    return res
