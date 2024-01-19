import time
import gymnasium as gym
import sapien
import mani_skill2.envs
import sapien.physx
import tqdm
import numpy as np
import sapien.render
import sapien.physx
if __name__ == "__main__":
    num_envs = 12
    env_id = "PickCube-v0"
    env = gym.make_vec(
        env_id,
        num_envs,
        vectorization_mode="async",
        vector_kwargs=dict(context="forkserver"),
    )
    np.random.seed(2022)
    env.reset(seed=2022)
    print("GPU Simulation Enabled:", sapien.physx.is_gpu_enabled())
    N = 1000
    stime = time.time()
    for i in tqdm.tqdm(range(N)):
        env.step(env.action_space.sample())
    dtime = time.time() - stime
    FPS = num_envs * N / dtime
    print(f"{FPS=:0.3f}. {N=} frames in {dtime:0.3f}s with {num_envs} parallel envs")
    N = 1000
    stime = time.time()
    for i in tqdm.tqdm(range(N)):
        actions = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(actions)
        if i % 200 == 0 and i != 0:
            env.reset()
            print("RESET")
    dtime = time.time() - stime
    FPS = num_envs * N / dtime
    print(f"{FPS=:0.3f}. {N=} frames in {dtime:0.3f}s with {num_envs} parallel envs with step+reset")
    env.close()