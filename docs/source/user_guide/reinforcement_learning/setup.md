# Setup

This page documents key things to know when setting up ManiSkill environments for reinforcement learning, including:

- How to convert ManiSkill environments to gymnasium API compatible environments, both [single](#gym-environment-api) and [vectorized](#gym-vectorized-environment-api) APIs.
- How to [**correctly** evaluate RL policies fairly](#evaluation)
- [Useful Wrappers](#useful-wrappers)

ManiSkill environments are created by gymnasium's `make` function. The result is by default a "batched" environment where every input and output is batched. Note that this is not standard gymnasium API. If you want the standard gymnasium environment / vectorized environment API see the next sections.

```python
import mani_skill.envs
import gymnasium as gym
N = 4
env = gym.make("PickCube-v1", num_envs=N)
env.action_space # shape (N, D)
env.observation_space # shape (N, ...)
env.reset()
obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
# obs (N, ...), rew (N, ), terminated (N, ), truncated (N, )
```

## Gym Environment API

If you want to use the CPU simulator / a single environment, you can apply the `CPUGymWrapper` which essentially unbatches everything and turns everything into numpy so the environment behaves just like a normal gym environment. The API for a gym environment is detailed on [their documentation](https://gymnasium.farama.org/api/env/).

```python
import mani_skill.envs
import gymnasium as gym
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
N = 1
env = gym.make("PickCube-v1", num_envs=N)
env = CPUGymWrapper(env)
env.action_space # shape (D, )
env.observation_space # shape (...)
env.reset()
obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
# obs (...), rew (float), terminated (bool), truncated (bool)
```

## Gym Vectorized Environment API

We adopt the gymnasium `VectorEnv` (also known as `AsyncVectorEnv`) interface as well and you can achieve that via a single wrapper so that your algorithms that assume `VectorEnv` interface can work seamlessly. The API for a vectorized gym environment is detailed on [their documentation](https://gymnasium.farama.org/api/vector/)

```python
import mani_skill.envs
import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
N = 4
env = gym.make("PickCube-v1", num_envs=N)
env = ManiSkillVectorEnv(env, auto_reset=True, ignore_terminations=False)
env.action_space # shape (N, D)
env.single_action_space # shape (D, )
env.observation_space # shape (N, ...)
env.single_observation_space # shape (...)
env.reset()
obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
# obs (N, ...), rew (N, ), terminated (N, ), truncated (N, )
```

You may also notice that there are two additional options when creating a vector env. The `auto_reset` argument controls whether to automatically reset a parallel environment when it is terminated or truncated. This is useful depending on algorithm. The `ignore_terminations` argument controls whether environments reset upon terminated being True. Like gymnasium vector environments, partial resets can occur where some parallel environments reset while others do not.

Note that for efficiency, everything returned by the environment will be a batched torch tensor on the GPU and not a batched numpy array on the CPU. This is the only difference you may need to account for between ManiSkill vectorized environments and gymnasium vectorized environments.

## Evaluation

With the number of different types of environments, algorithms, and approaches to evaluation, we describe below a consistent and standardized way to evaluate all kinds of policies in ManiSkill fairly. In summary, the following setup is necessary to ensure fair evaluation:

- Partial resets are turned off and environments do not reset upon success/fail/termination (`ignore_terminations=True`). Instead we record multiple types of success/fail metrics.
- All parallel environments reconfigure on reset (`reconfiguration_freq=1`), which randomizes object geometries if the task has object randomization.


The code to fairly evaluate policies and record standard metrics in ManiSkill are shown below. For GPU vectorized environments the code to evaluate policies by environment ID the following is recommended:

```python
import gymnasium as gym
import torch
from collections import defaultdict
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
env_id = "PushCube-v1"
num_eval_envs = 64
env_kwargs = dict(obs_mode="state") # modify your env_kwargs here
eval_envs = gym.make(env_id, num_envs=num_eval_envs, reconfiguration_freq=1, **env_kwargs)
# add any other wrappers here
eval_envs = ManiSkillVectorEnv(eval_envs, ignore_terminations=True, record_metrics=True)

# evaluation loop, which will record metrics for complete episodes only
obs, _ = eval_envs.reset(seed=0)
eval_metrics = defaultdict(list)
for _ in range(400):
    action = eval_envs.action_space.sample() # replace with your policy action
    obs, rew, terminated, truncated, info = eval_envs.step(action)
    # note as there are no partial resets, truncated is True for all environments at the same time
    if truncated.any():
        for k, v in info["final_info"]["episode"].items():
            eval_metrics[k].append(v.float())
for k in eval_metrics.keys():
    print(f"{k}_mean: {torch.mean(torch.stack(eval_metrics[k])).item()}")
```

And for CPU vectorized environments the following is recommended for evaluation:

```python
import gymnasium as gym
import numpy as np
from collections import defaultdict
from mani_skill.utils.wrappers import CPUGymWrapper
env_id = "PickCube-v1"
num_eval_envs = 8
env_kwargs = dict(obs_mode="state") # modify your env_kwargs here
def cpu_make_env(env_id, env_kwargs = dict()):
    def thunk():
        env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
        # add any other wrappers here
        return env
    return thunk

if __name__ == "__main__":
    vector_cls = gym.vector.SyncVectorEnv if num_eval_envs == 1 else lambda x : gym.vector.AsyncVectorEnv(x, context="forkserver")
    eval_envs = vector_cls([cpu_make_env(env_id, env_kwargs) for _ in range(num_eval_envs)])
    # evaluation loop, which will record metrics for complete episodes only
    obs, _ = eval_envs.reset(seed=0)
    eval_metrics = defaultdict(list)
    for _ in range(400):
        action = eval_envs.action_space.sample() # replace with your policy action
        obs, rew, terminated, truncated, info = eval_envs.step(action)
        # note as there are no partial resets, truncated is True for all environments at the same time
        if truncated.any():
            for final_info in info["final_info"]:
                for k, v in final_info["episode"].items():
                    eval_metrics[k].append(v)
    for k in eval_metrics.keys():
        print(f"{k}_mean: {np.mean(eval_metrics[k])}")
```

The following metrics are recorded and explained below:
- `success_once`: Whether the task was successful at any point in the episode.
- `success_at_end`: Whether the task was successful at the final step of the episode.
- `fail_once/fail_at_end`: Same as the above two but for failures. Note not all tasks have success/fail criteria.
- `return`: The total reward accumulated over the course of the episode.

## Useful Wrappers

RL practitioners often use wrappers to modify and augment environments. These are documented in the [wrappers](../wrappers/index.md) section. Some commonly used ones include:
- [RecordEpisode](../wrappers/record.md) for recording videos/trajectories of rollouts.
- [FlattenRGBDObservations](../wrappers/flatten.md#flatten-rgbd-observations) for flattening the `obs_mode="rgbd"` or `obs_mode="rgb+depth"` observations into a simple dictionary with just a combined `rgbd` tensor and `state` tensor.

## Common Mistakes / Gotchas

In old environments/benchmarks, people often have used `env.render(mode="rgb_array")` or `env.render()` to get image inputs for RL agents. This is not correct because image observations are returned by `env.reset()` and `env.step()` directly and `env.render` is just for visualization/video recording only in ManiSkill.

For robotics tasks observations often are composed of state information (like robot joint angles) and image observations (like camera images). All tasks in ManiSkill will specifically remove certain privileged state information from the observations when the `obs_mode` is not `state` or `state_dict` like ground truth object poses. Moreover, the image observations returned by `env.reset()` and `env.step()` are usually from cameras that are positioned in specific locations to provide a good view of the task to make it solvable.
