import gc

import psutil
import sapien

import mani_skill.envs

# sapien.set_log_level("info")
"""
Notes:
Not deleting sapien entities is faster to reconfigure and does not cause memory issue on (cpu, state)


- GPU/CPU, State/RGBD, no reconfigure = stable
- CPU, State, Reconfigure = Mostly no memory increase (entities are getting deleted). It increases very slowly at seemingly random intervals, unclear how
- GPU, State, Reconfigure = Fluctuates, increases slowly (entities are getting deleted). Could be same issue as ^

- CPU, Sensor data, Reconfigure = increases very slowly
- GPU, Sensor data, Reconfigure = Increases fairly slowly
"""
if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make(
        "PushCube-v1",
        obs_mode="state",
        num_envs=22,
    )
    env.reset(seed=0)
    process = psutil.Process()
    print(f"{process.memory_info().rss / 1000 / 1000:0.02f}MB")
    i = 0
    while True:
        i += 1
        env.step(None)
        if i % 10 == 0:
            gc.collect()
            env.reset(options=dict(reconfigure=True))
            print(f"{i}, {process.memory_info().rss / 1000 / 1000:0.02f}MB")
