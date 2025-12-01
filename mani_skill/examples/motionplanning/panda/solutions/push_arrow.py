import numpy as np
import sapien
import gymnasium as gym
import torch
import time
from transforms3d.euler import euler2quat
from mani_skill.envs.tasks import PushArrowEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
from copy import deepcopy
def main():
    env: PushArrowEnv = gym.make(
        "PushArrow-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
    )
    for seed in range(100):
        res = solve(env, seed=seed, debug=False, vis=True)
        print(res)
    env.close()

def solve(env: PushArrowEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )

    env = env.unwrapped

    # Get tool OBB and compute grasp pose
    obb = get_actor_obb(env.arrow)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=0.03,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.arrow.pose.sp.p)
    offset = sapien.Pose([0, 0, 0])
    grasp_pose = grasp_pose * (offset)
    reach_pose_1 = grasp_pose
    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    res = None
    arrow_transform = env.arrow.pose.to_transformation_matrix()[0]
    arrow_x_axis = arrow_transform[:3, 0]  # First column is X axis
    arrow_x_axis = arrow_x_axis / np.linalg.norm(arrow_x_axis)
    arrow_init_x_axis = deepcopy(arrow_x_axis)

    while(np.dot(arrow_x_axis, arrow_init_x_axis) > -0.95): # while angle < 170 degrees
        print("Current dot product:", np.dot(arrow_x_axis, arrow_init_x_axis))
        # z_rot = env.quat_to_z_euler(env.arrow.pose.q)
        # reach_pose_1 = reach_pose_1 * sapien.Pose([0, 0.01, 0]) # 15 cm above grasp pose
        # reach_pose_1.set_q(euler2quat(np.pi,0,z_rot)) # set orientation to arrow orientation
        # print(torch.tensor(reach_pose_1.q).reshape(1,4)[:, -1], torch.tensor(env.arrow.pose.q)[:, -1])
        # res = planner.move_to_pose_with_RRTConnect(reach_pose_1)
        # if res == -1: return res

        # Get arrow's current transformation matrix
        arrow_transform = env.arrow.pose.to_transformation_matrix()[0]
        arrow_x_axis = arrow_transform[:3, 0]  # First column is X axis
        
        # Create rotation matrix where X axis matches arrow's X axis
        # and Z axis points down (for the gripper)
        z_axis = np.array([0, 0, -1])  # gripper points down
        y_axis = np.cross(z_axis, arrow_x_axis)  # compute Y axis
        y_axis = y_axis / np.linalg.norm(y_axis)  # normalize
        x_axis = np.cross(y_axis, z_axis)  # recompute X to ensure orthogonality
        
        # Build rotation matrix
        R = np.column_stack([x_axis, y_axis, z_axis])
        
        # Convert to quaternion (you may need to import)
        from transforms3d.quaternions import mat2quat
        q = mat2quat(R)
        
        # Set pose
        reach_pose_1.set_q(q)
        # Translate in local Y (will follow new orientation)
        reach_pose_1 = reach_pose_1 * sapien.Pose([0.001, 0.01, 0])
        res = planner.move_to_pose_with_RRTConnect(reach_pose_1)
        if res == -1: return res
    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    # planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift tool to safe height
    # -------------------------------------------------------------------------- #
    # lift_height = 0.35  
    # lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]))
    # lift_pose.set_q(grasp_pose.q)  # Maintain grasp orientation
    # res = planner.move_to_pose_with_screw(lift_pose)
    # if res == -1: return res

    # ball_pos = env.ball.pose.sp.p
    # approach_offset = sapien.Pose(
    #     [-(env.hook_length + env.cube_half_size + 0.08),  
    #     -0.0,  
    #     lift_height - 0.05]  
    # )
    # approach_pose = sapien.Pose(cube_pos) * approach_offset
    # approach_pose.set_q(grasp_pose.q)
    
    # res = planner.move_to_pose_with_screw(approach_pose)
    # if res == -1: return res

    # # -------------------------------------------------------------------------- #
    # # Lower tool behind cube
    # # -------------------------------------------------------------------------- #
    # behind_offset = sapien.Pose(
    #     [-(env.hook_length + env.cube_half_size),  
    #     -0.067,  
    #     0] 
    # )
    # hook_pose = sapien.Pose(cube_pos) * behind_offset
    # hook_pose.set_q(grasp_pose.q)
    
    # res = planner.move_to_pose_with_screw(hook_pose)
    # if res == -1: return res

    # # -------------------------------------------------------------------------- #
    # # Pull cube
    # # -------------------------------------------------------------------------- #
    # pull_offset = sapien.Pose([-0.35, 0, 0])
    # target_pose = hook_pose * pull_offset
    # res = planner.move_to_pose_with_screw(target_pose)
    # if res == -1: return res
    # print("\nLift complete. Holding position for 5 seconds...")
    # time.sleep(5)
    planner.close()
    return res


if __name__ == "__main__":
    main()
