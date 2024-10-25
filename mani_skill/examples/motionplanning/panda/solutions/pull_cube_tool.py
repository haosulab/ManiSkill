import numpy as np
import sapien

from mani_skill.envs.tasks import PullCubeToolEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb

def solve(env: PullCubeToolEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    env = env.unwrapped
    
    # Get tool OBB for grasping
    tool_obb = get_actor_obb(env.l_shape_tool)
    
    # Define grasp approach for the tool
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    # Compute grasp pose for the tool
    grasp_info = compute_grasp_info_by_obb(
        tool_obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=0.02,  # Finger penetration depth
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.l_shape_tool.pose.sp.p)

    # Pre-grasp pose slightly above the tool
    pre_grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(pre_grasp_pose)
    
    # Move to grasp pose and close gripper
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # Calculate position behind the cube for hooking
    cube_pos = env.cube.pose.sp.p
    hook_offset = np.array([-env.hook_length - env.cube_half_size, 0, 0])
    hook_pose = sapien.Pose(cube_pos + hook_offset, grasp_pose.q)
    
    # Move tool behind cube
    planner.move_to_pose_with_screw(hook_pose)
    
    # Calculate target position within arm's workspace
    workspace_center = np.array([env.arm_reach * 0.7, 0, cube_pos[2]])
    target_pose = sapien.Pose(workspace_center, grasp_pose.q)
    
    # Pull cube to target position
    res = planner.move_to_pose_with_screw(target_pose)

    planner.close()
    return res