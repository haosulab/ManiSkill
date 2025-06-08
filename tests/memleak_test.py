import gc

import gymnasium as gym
import psutil

import mani_skill.envs


def main():
    process = psutil.Process()
    print(f"{process.memory_info().rss / 1000 / 1000:0.02f}MB")
    env = gym.make("PickCube-v1", num_envs=64, obs_mode="none", reward_mode="sparse")
    env.reset(seed=0)
    i = 0
    while True:
        env.unwrapped.scene.px.cuda_rigid_body_data  # this is fine, no increase in mem
        env.unwrapped.scene.px.cuda_rigid_body_data.torch()  # this increases memory
        i += 1
        if i % 100 == 0:
            gc.collect()
            print(f"{i}, {process.memory_info().rss / 1000 / 1000:0.02f}MB")


if __name__ == "__main__":
    main()
