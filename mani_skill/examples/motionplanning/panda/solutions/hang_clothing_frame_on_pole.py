import gymnasium as gym
import numpy as np
import sapien
import math
from mani_skill.envs.tasks.tabletop.hang_clothing_frame_on_pole import HangClothingFrameOnPoleEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
import time
def main():
    env: HangClothingFrameOnPoleEnv = gym.make(
        "HangClothingFrameOnPole-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        )
    for seed in range(100):
        res = solve(env, seed=seed, debug=True, vis=True)
        print(res)
    env.close()

def solve(env: HangClothingFrameOnPoleEnv, seed=None, debug=False, vis=False):
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
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )
    
    env = env.unwrapped
    FINGER_LENGTH = 0.025
    obb = get_actor_obb(env.clothing_frame)

    approaching = np.array([0, -1/math.sqrt(2), -1/math.sqrt(2)])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    frame_init_pos = env.clothing_frame.pose

    # init_pose = book_init_pos * sapien.Pose([-0.5, 0, 0.0])
    # res = planner.move_to_pose_with_screw(init_pose)
    # if res == -1: return res



    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)    
    grasp_pose = grasp_pose*sapien.Pose([0, 0, 0.1])  # slightly above the book center

    # ------------------------------------------------------------------
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0.0, 0.0, -0.1])
    res = planner.move_to_pose_with_RRTStar(reach_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_RRTStar(grasp_pose)
    if res == -1: return res
    planner.close_gripper(gripper_state=-0.6)

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = grasp_pose*sapien.Pose([0.0, 0.0, -0.3])
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Rotation about the z axis by 180 (world coordinates)
    # -------------------------------------------------------------------------- #
    # theta = np.pi/2
    theta = np.pi
    R_world_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    T_current = lift_pose.to_transformation_matrix()
    T_rotated = np.eye(4)
    T_rotated[:3, :3] = R_world_z @ T_current[:3, :3]
    T_rotated[:3, 3] = T_current[:3, 3]
    final_pose = sapien.Pose(matrix=T_rotated)

    res = planner.move_to_pose_with_RRTStar(final_pose)
    if res == -1: return res
    # -------------------------------------------------------------------------- #
    # Rotation about the y axis
    # -------------------------------------------------------------------------- #
    # rotation_quat = np.array([0, 0, np.cos(theta / 2), np.sin(theta / 2)]) 
    
    # final_pose = lift_pose * sapien.Pose(
    #     p=[0, 0, 0],
    #     q=rotation_quat
    # )
    # res = planner.move_to_pose_with_screw(final_pose)
    # if res == -1: return res
    # -------------------------------------------------------------------------- #
    # Rotation about the y axis and translation along the local z axis
    # -------------------------------------------------------------------------- #
    # theta = -np.pi/2lander
    # rotation_quat = np.array([np.cos(theta / 2), np.sin(theta / 2), 0.0 , 0.0])
    # final_pose = final_pose * sapien.Pose(
    #     p=[0, 0, 0],
    #     q=rotation_quat
    # ) * sapien.Pose([0, 0, -0.10])
    final_pose = sapien.Pose([0.0, 0.1, -0.13])*final_pose
    res = planner.move_to_pose_with_RRTStar(final_pose)
    if res == -1: return res
    # # -------------------------------------------------------------------------- #
    # # Lower
    # # -------------------------------------------------------------------------- #
    final_pose = sapien.Pose([0,0.4,0])*final_pose
    res = planner.move_to_pose_with_RRTStar(final_pose)
    if res == -1: return res

    planner.open_gripper()
    res = planner.move_to_pose_with_screw(final_pose)
    if res == -1: return res

    # Retreat
    retreat_pose = sapien.Pose([0, -0.4,0])*final_pose
    res = planner.move_to_pose_with_RRTStar(retreat_pose)
    if res == -1: return res
    planner.close()
    
    return res

if __name__ == "__main__":
    main()
