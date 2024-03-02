import sys

import gymnasium as gym

import mani_skill2.envs
from mani_skill2.envs.sapien_env import BaseEnv

if __name__ == "__main__":
    robot = sys.argv[1]
    # robot in ["panda", "fetch", "xmate3_robotiq"]:
    env = gym.make(
        "Empty-v1",
        enable_shadow=True,
        robot_uids=robot,
        render_mode="human",
        control_mode=None,
        shader_dir="rt-fast",
    )
    env: BaseEnv = env.unwrapped
    viewer = env.render()
    viewer.paused = True

    while True:
        viewer = env.render()
        # if viewer.window.key_press("n"):
        #     env.close()
        #     del env
        #     break
