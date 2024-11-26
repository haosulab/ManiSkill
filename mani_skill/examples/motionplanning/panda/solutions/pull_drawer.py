import gymnasium as gym
import numpy as np
import sapien
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb

def main():
    env = gym.make(
        "PullDrawer-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="sparse"
    )
    
    for seed in range(100):
        res = solve(env, seed=seed, debug=False, vis=True)
        print(res[-1])
    env.close()

def solve(env, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in ["pd_joint_pos", "pd_joint_pos_vel"]
    
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )
    
    env = env.unwrapped
    FINGER_LENGTH = 0.025

    # Get handle position and compute grasp
    handle_pos = [-env.inner_width/2 - env.handle_offset, 0, 0]
    handle_pose = env.drawer_link.pose * sapien.Pose(p=handle_pos)
    
    # Compute grasp pose for handle
    approaching = np.array([1, 0, 0])  # Approach from front
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[:3, 1]
    
    handle_obb = {
        'center': handle_pose.p,
        'extents': np.array([env.handle_thickness, env.handle_width, env.handle_height]) * 2,
        'rotation': handle_pose.to_transformation_matrix()[:3, :3]
    }
    
    grasp_info = compute_grasp_info_by_obb(
        handle_obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH
    )
    
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # -------------------------------------------------------------------------- #
    # Pre-grasp pose
    # -------------------------------------------------------------------------- #
    pre_grasp_pose = grasp_pose * sapien.Pose([0.05, 0, 0])
    res = planner.move_to_pose_with_screw(pre_grasp_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Grasp handle
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Pull drawer
    # -------------------------------------------------------------------------- #
    pull_distance = env.max_pull_distance * 0.8
    steps = 5
    
    # Pull in small increments with refinement
    for i in range(steps):
        current_pull = (i + 1) * pull_distance / steps
        pull_pose = grasp_pose * sapien.Pose([-current_pull, 0, 0])
        
        # Move and refine position
        res = planner.move_to_pose_with_screw(pull_pose, refine_steps=3)
        if res == -1: return res
        
        # Check if drawer is actually moving
        if not env.agent.is_grasping(env.drawer_link):
            return res

    planner.close()
    return res

if __name__ == "__main__":
    main()
