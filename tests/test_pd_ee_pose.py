from time import time

import gymnasium as gym
import numpy as np
import pytest
import torch

import mani_skill.envs


@pytest.mark.gpu_sim
def test_pd_ee_pose_controller():
    nenvs = 64
    env = gym.make(
        "PickCube-v1",
        num_envs=nenvs,
        obs_mode="state",
        control_mode="pd_ee_pose",
        # render_mode="human",
        sim_backend="gpu",
    )
    torch.set_printoptions(linewidth=500)
    env.reset()
    target_pose = torch.tensor([0.4, 0.1, 0.5, np.pi / 2, 0.0, 0.0, 10.0]).expand(
        nenvs, 7
    )
    t0 = time()
    for i in range(100):
        env.step(target_pose)
        # env.render()
        print(f"Step {i}")
    print(f"Time taken: {time() - t0}")
    env.close()
    print("Done")


if __name__ == "__main__":
    test_pd_ee_pose_controller()
