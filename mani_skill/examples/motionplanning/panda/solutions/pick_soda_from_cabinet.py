import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.pick_soda_from_cabinet import PickSodaFromCabinetEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb

def main():
    env: PickSodaFromCabinetEnv = gym.make(
        "PickSodaFromCabinet-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
    )
    for seed in range(100):
        res = solve(env, seed=seed, debug=True, vis=True)
        print(res)
    env.close()

def solve(env: PickSodaFromCabinetEnv, seed=None, debug=False, vis=False):
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
    obb = get_actor_obb(env.soda)

    approaching = np.array([1, 0, 0])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    soda_init_pos = env.soda.pose

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
    grasp_pose = grasp_pose*sapien.Pose([0, 0, 0])  # slightly above the book center

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0.0, 0.0, -0.3])
    res = planner.move_to_pose_with_RRTStar(reach_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_RRTConnect(grasp_pose)
    if res == -1: return res
    planner.close_gripper(gripper_state=-0.6)

    # -------------------------------------------------------------------------- #
    # Move Back
    # -------------------------------------------------------------------------- #
    back_pose = grasp_pose * sapien.Pose([0, 0, -0.3])
    res = planner.move_to_pose_with_RRTStar(back_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Rotation about the x axis
    # -------------------------------------------------------------------------- #
    # theta = np.pi/2
    # rotation_quat = np.array([0.5, -0.5, 0.5, 0.5])  
    
    # final_pose = lift_pose * sapien.Pose(
    #     p=[0, 0, 0],
    #     q=rotation_quat
    # )
    # # For such complex motions it is better to use RRTStar
    # res = planner.move_to_pose_with_RRTStar(final_pose)
    # if res == -1: return res
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
    # theta = -np.pi/2
    # rotation_quat = np.array([np.cos(theta / 2), np.sin(theta / 2), 0.0 , 0.0])
    # final_pose = final_pose * sapien.Pose(
    #     p=[0, 0, 0],
    #     q=rotation_quat
    # ) * sapien.Pose([0, 0, -0.10])
    final_pose = sapien.Pose(p=[-0.053, -0.160, 0.1],q=grasp_pose.q)
    res = planner.move_to_pose_with_RRTStar(final_pose)
    if res == -1: return res
    # -------------------------------------------------------------------------- #
    # Lower
    # -------------------------------------------------------------------------- #
    # lower_pose = final_pose * sapien.Pose([0, 0, 0.2])
    # res = planner.move_to_pose_with_RRTStar(lower_pose)
    # if res == -1: return res

    planner.open_gripper()
    res = planner.move_to_pose_with_screw(final_pose)
    if res == -1: return res

    planner.close()
    
    return res

if __name__ == "__main__":
    main()
