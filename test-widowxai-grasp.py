"""
You should double check if the is_grasping function is working as intended. It depends on the orientation of the gripper links and some robots have this orientation flipped compared to others. A way to check is to create the pick cube env, use pd_joint_pos action space, and script the joint positions so that the robot moves to the cube, then tries to grasp it, and then check the returned info object for whether the env thinks the robot is grasping it.    

to get these joint pos reference points, you can open the viewer, and click the end-effector and click the transform tab to drag the robot to the cube and figure out where it is.
"""
import time
import gymnasium as gym
import numpy as np

import mani_skill.envs

env = gym.make("PickCube-v1", robot_uids="widowxai", control_mode="pd_joint_pos")
obs, info = env.reset(seed=0)
num_steps_per_pose = 200
joint_positions = {
    "pre-grasp" : np.array([0.096, 2.647, 2.335, -0.738, 0.0, 0.0, 0.04]),
    "grasp" : np.array([0.096, 2.647, 2.335, -0.738, 0.0, 0.0, 0.02]),
}

for position_name, joint_pos in joint_positions.items():
    print(f"\n--- Moving to position: {position_name} ---")
    for step in range(num_steps_per_pose):
        obs, rew, term, trunc, info = env.step(joint_pos)
        env.render_human()
        print(f"is_grasped: {info['is_grasped']}, success: {info['success']}")

env.close()