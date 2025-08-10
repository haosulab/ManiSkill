import gymnasium as gym
import mani_skill.envs
import time
import numpy as np

env = gym.make("PickCube-v1",
               render_mode = "human",
               robot_uids = "xarm6_robotiq_140")
obs, _ = env.reset(seed=0)
env.unwrapped.print_sim_details() # print verbose details about the configuration
done = False
start_time = time.time()
t = 0
while not done:
    # action = np.random.randn(1)
    # print(t)
    t += 1
    if t < 100:
        action = np.array([0,0,0,0,0,0,-0.05])
    else:
        action = np.array([0,0,0.1,0,0,0,-0.05])
        # import pdb;pdb.set_trace()
    obs, rew, terminated, truncated, info = env.step(action)
    # done = terminated or truncated
    done = False
    env.render()
    
N = info["elapsed_steps"].item()
dt = time.time() - start_time   
FPS = N / (dt)
print(f"Frames Per Second = {N} / {dt} = {FPS}")