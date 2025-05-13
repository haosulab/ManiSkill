# Setup

This page documents key things to know when setting up ManiSkill environments for learning from demonstrations, including:

- How to [download and replay trajectories to standard datasets](#downloading-and-replaying-trajectories--standard-datasets) used for benchmarking state-based and vision-based imitation learning
- How to [evaluate trained models fairly and correctly](#evaluation)
- Some common [pitfalls to avoid](#common-pitfalls-to-avoid)


## Downloading and Replaying Trajectories / Standard Datasets

By default for fast downloads and smaller file sizes, ManiSkill demonstrations are stored in a highly reduced/compressed format which includes not keeping any observation data. Run the command to download the raw minimized demonstration data.

```bash
python -m mani_skill.utils.download_demo "PickCube-v1"
```

To ensure everyone has the same preprocessed/replayed dataset, make sure to run this script: [https://github.com/haosulab/ManiSkill/blob/main/scripts/data_generation/replay_for_il_baselines.sh](https://github.com/haosulab/ManiSkill/blob/main/scripts/data_generation/replay_for_il_baselines.sh). Note that some scripts use the GPU simulation (marked by `-b physx_cuda`) for replay, which may potentially require more GPU memory than you have available. You can always lower the number of parallel environments used to be replayed by setting `--num-procs` to a lower number.

It has fixed settings for the trajectory replay to generate observation data and set the desired action space/controller for all benchmarked tasks. All benchmarked results in the [Wandb project detailing all benchmarked training runs](https://wandb.ai/stonet2000/ManiSkill) use the data replayed by the script above

If you need more advanced use-cases for trajectory replay (e.g. generating pointclouds, changing controller modes), see the [trajectory replay documentation](../datasets/replay.md). If you want to generate the original datasets yourself locally we save all scripts used for dataset generation in the [data_generation](https://github.com/haosulab/ManiSkill/tree/main/scripts/data_generation) folder.


## Evaluation

With the number of different types of environments, algorithms, and approaches to evaluation, we describe below a consistent and standardized way to evaluate all kinds of learning from demonstrations policies in ManiSkill fairly. In summary, the following setup is necessary to ensure fair evaluation:

- Partial resets are turned off and environments do not reset upon success/fail/termination (`ignore_terminations=True`). Instead we record multiple types of success/fail metrics.
- All parallel environments reconfigure on reset (`reconfiguration_freq=1`), which randomizes object geometries if the task has object randomization.

The code to fairly evaluate policies and record standard metrics in ManiSkill are shown below. We provide CPU and GPU vectorized options particularly because depending on what simulation backend your demonstration data is collected on you will want to evaluate your policy on the same backend.

For GPU vectorized environments the code to evaluate policies by environment ID the following is recommended:

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

<!-- NOTE (stao): the content for evaluation is the same as in the RL setup.md document, however I don't really want users have to click to a separate page to learn about evaluation... -->

Generally for learning from demonstrations the only metric that matters is "success_once" and is what is typically reported in research/work using ManiSkill.


## Common Pitfalls to Avoid

In general if demonstrations are collected in e.g. the PhysX CPU simulation, you want to ensure you evaluate any policy trained on that data in the same simulation backend. For highly precise tasks (e.g. PushT) where even a 1e-3 error can lead to different results, this is especially important. This is why all demonstrations replayed by our trajectory replay tool will annotate the simulation backend used on the trajectory file name

Your source of demonstration data can largely affect the training performance. Classic behavior cloning can do decently well to imitate demonstrations generated by a neural network / RL trained policy, but will struggle to imitate more multi-modal demonstrations (e.g. human teleoperated or motion planning generated). Methods like Diffusion Policy (DP) are designed to address this problem. If you are unsure, all official datasets from ManiSkill will detail clearly in the trajectory metadata JSON file how the data was collected and type of data it is.