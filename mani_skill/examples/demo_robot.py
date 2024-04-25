"""
Instantiates a empty environment with a floor, and attempts to place any given robot in there
"""

import sys

import gymnasium as gym

import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv

if __name__ == "__main__":
    robot = sys.argv[1]
    # robot in ["panda", "fetch", "xmate3_robotiq"]:
    env = gym.make(
        "Empty-v1",
        enable_shadow=True,
        robot_uids=robot,
        render_mode="human",
        # control_mode="arm_pd_ee_delta_pose_gripper_pd_joint_pos",
        # shader_dir="rt-fast",
    )
    env.reset(seed=0)
    env: BaseEnv = env.unwrapped
    viewer = env.render()
    viewer.paused = True

    while True:
        env.step(env.action_space.sample())
        viewer = env.render()
        # if viewer.window.key_press("n"):
        #     env.close()
        #     del env
        #     break
