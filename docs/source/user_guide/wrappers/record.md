# Recording Episodes

ManiSkill provides a few ways to record videos/trajectories of tasks on single and vectorized environments. The recommended way is via [a wrapper](#recordepisode-wrapper). The other way is to [call a function](#capture-individual-images) to generate the video frames yourself and compile them into a video yourself.

## RecordEpisode Wrapper

The recommended approach is to use our RecordEpisode wrapper, which supports both single and vectorized environments, and saves videos and/or trajectory data (in the [ManiSkill format](../datasets/demos.md)) to disk. It will save whatever render_mode is specified upon environment creation (can be "rgb_array", "sensors", or "all" which combines both).

This wrapper by default saves videos on environment reset for single environments.
```python
import mani_skill.envs
import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode
env = gym.make("PickCube-v1", num_envs=1, render_mode="rgb_array")
env = RecordEpisode(env, output_dir="videos", save_trajectory=True, trajectory_name="trajectory", save_video=True, video_fps=30)
env.reset()
for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        env.reset()
```

For vectorized environments, the wrapper will save videos of length `max_steps_per_video` before flushing the video to disk and starting a new video. It does not save on reset as environments can have partial resets.

```python
import mani_skill.envs
import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
N = 4
env = gym.make("PickCube-v1", num_envs=N, render_mode="rgb_array")
env = RecordEpisode(env, output_dir="videos", save_trajectory=True, trajectory_name="trajectory", max_steps_per_video=50, video_fps=30)
env = ManiSkillVectorEnv(env, auto_reset=True) # adds auto reset
env.reset()
for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Capture Individual Images

If you want to use your own custom video recording methods, then you can call the API directly to capture images of the environment. This works the same for both single and vectorized environments.

```python
import mani_skill.envs
import gymnasium as gym
N = 1
env = gym.make("PickCube-v1", num_envs=N)
images = []
env.reset()
images.append(env.render_rgb_array())
for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    images.append(env.render_rgb_array())
    # env.render_sensors() # render sensors mode
    # env.render_all() # render all mode
```

Note that the return of `env.render_rgb_array(), env.render_sensors()` etc. are all batched torch tensors on the GPU. You will likely need to convert them to CPU numpy arrays to save them to disk.