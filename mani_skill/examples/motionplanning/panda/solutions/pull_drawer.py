import gymnasium as gym
import numpy as np
import sapien
import torch

from mani_skill.envs.tasks import PullDrawerEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.utils.structs import Pose


def solve(env: PullDrawerEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )
    
    # Get drawer handle position
    # planner.open_gripper()

    drawer_link = env.drawer.links_map['drawer']


    handle_offset = np.array([-env.inner_width/2 - env.handle_offset, 0, 0])
    handle_pose = drawer_link.pose * sapien.Pose(handle_offset)

    
    # Pre-grasp pose slightly behind handle
    approach_offset = sapien.Pose(p = [-0.1, 0, 0])  # Offset behind handle
    pre_grasp_pose = handle_pose * approach_offset
    # First 90-degree rotation around x-axis, then 90-degree rotation around z-axis
    pre_grasp_pose.q = torch.tensor([0.5, 0.5, 0.5, 0.5])

    res = planner.move_to_pose_with_screw(pre_grasp_pose)
    

    move_in_offset = sapien.Pose(p = [0, 0, 0.13])
    grasp_pose = pre_grasp_pose * move_in_offset

    res = planner.move_to_pose_with_screw(grasp_pose)


    if res == -1:
        return res
    
    # Close gripper
    planner.close_gripper()

    # Pull drawer
    pull_offset = sapien.Pose([0, 0, -env.max_pull_distance * 0.81])
    pull_pose = grasp_pose * pull_offset
    res = planner.move_to_pose_with_screw(pull_pose)
    
    planner.close()
    return res

def main():
    env = gym.make(
        "PullDrawer-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array"
        sim
    )
    
    for seed in range(100):
        res = solve(env, seed=seed, debug=False, vis=True)
        print(f"Seed {seed}: {res}")
        
    env.close()

if __name__ == "__main__":
    main()
