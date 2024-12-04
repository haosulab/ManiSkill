import gymnasium as gym
import numpy as np
import sapien
import torch

from mani_skill.envs.tasks import PullDrawerEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.utils.structs import Pose


def main():
    env: PullDrawerEnv = gym.make(
        "PullDrawer-v1",
        obs_mode="none", 
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
    )
    
    for seed in range(100):
        res = solve(env, seed=seed, debug=False, vis=True)
        print(f"Seed {seed} result: {res}")
    
    env.close()


def solve(env: PullDrawerEnv, seed=None, debug=False, vis=False):
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
    
    # Compute drawer handle pose and dimensions
    drawer_pose = env.drawer.pose
    handle_offset = np.array([-env.inner_width/2 - env.handle_offset, 0, 0])
    handle_pose = drawer_pose.to_transformation_matrix()[0, :3, :3] @ handle_offset + drawer_pose.p
    
    # Define grasp approach and closing direction
    approaching = np.array([1, 0, 0])  # Approach from side of handle
    tcp_forward = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1]
    
    # Compute grasp pose
    grasp_center = handle_pose
    grasp_offset = 0.05  # Small offset to account for gripper width
    
    # Create initial approach and grasp poses
    approach_pose = Pose.create_from_pq(
        p=grasp_center + approaching * grasp_offset,
        q=env.agent.tcp.pose.q
    )
    
    grasp_pose = Pose.create_from_pq(
        p=grasp_center,
        q=env.agent.tcp.pose.q
    )
    
    # Motion planning stages
    # -------------------------------------------------------------------------- #
    # Stage 1: Approach Handle
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(approach_pose)
    if res == -1:
        return res
    
    # -------------------------------------------------------------------------- #
    # Stage 2: Grasp Handle
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        return res
    
    planner.close_gripper()
    
    # -------------------------------------------------------------------------- #
    # Stage 3: Pull Drawer
    # -------------------------------------------------------------------------- #
    # Define pull trajectory
    pull_distance = env.max_pull_distance * 0.8  # Pull 80% of max distance
    pull_direction = np.array([-1, 0, 0])  # Pull along drawer's movement axis
    
   
    for fraction in [0.25, 0.5, 0.75, 1.0]:
        pull_pose = Pose.create_from_pq(
            p=grasp_center + pull_direction * (pull_distance * fraction),
            q=grasp_pose.q
        )
        
        res = planner.move_to_pose_with_screw(pull_pose)
        if res == -1:
            return res
    
    # Final stage: release and verify
    planner.open_gripper()
    
    return 0


if __name__ == "__main__":
    main()
