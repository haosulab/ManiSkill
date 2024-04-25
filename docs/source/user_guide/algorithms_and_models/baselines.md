# Baselines

ManiSkill has a number of baseline Reinforcement Learning (RL), Learning from Demonstrations (LfD) / Imitation Learning (IL) algorithms implemented that are easily runnable and reproducible for ManiSkill tasks. All baselines have their own standalone folders that you can download and run the code without having. The tables in the subsequent sections list out the implemented baselines, where they can be found, as well as results of running that code with tuned hyperparameters on some relevant ManiSkill tasks.

<!-- TODO: Add pretrained models? -->

<!-- Acknowledgement: This neat categorization of algorithms is taken from https://github.com/tinkoff-ai/CORL -->

## Offline Only Methods
These are algorithms that do not use online interaction with the task to be trained and only learn from demonstration data. 
<!-- Note that some of these algorithms can be trained offline and online and are marked with a \* and discussed in a [following section](#offline--online-methods) -->

| Baseline                                                   | Source                                                                                             | Results               |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------- |
| Behavior Cloning                                           | [source](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/behavior-cloning)     | [results](#baselines) |
| [Decision Transformer](https://arxiv.org/abs/2106.01345)   | [source](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/decision-transformer) | [results](#baselines) |
| [Decision Diffusers](https://arxiv.org/abs/2211.15657.pdf) | [source](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/decision-diffusers)   | [results](#baselines) |


## Online Only Methods
These are online only algorithms that do not learn from demonstrations and optimize based on feedback from interacting with the task. These methods also benefit from GPU simulation which can massively accelerate training time

| Baseline                                                               | Source                                                                             | Results               |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | --------------------- |
| [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) | [source](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/ppo)  | [results](#baselines) |
| [Soft Actor Critic (SAC)](https://arxiv.org/abs/1801.01290)            | [source](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/sac)  | [results](#baselines) |
| [REDQ](https://arxiv.org/abs/2101.05982)                               | [source](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/redq) | [results](#baselines) |


## Offline + Online Methods
These are baselines that can train on offline demonstration data as well as use online data collected from interacting with an task.

| Baseline                                                                                  | Source                                                                              | Results               |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | --------------------- |
| [Soft Actor Critic (SAC)](https://arxiv.org/abs/1801.01290) with demonstrations in buffer | [source](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/sac)   | [results](#baselines) |
| [MoDem](https://arxiv.org/abs/2212.05698)                                                 | [source](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/modem) | [results](#baselines) |
| [RLPD](https://arxiv.org/abs/2302.02948)                                                  | [source](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/rlpd)  | [results](#baselines) |


