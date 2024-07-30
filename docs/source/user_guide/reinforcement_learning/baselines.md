# Baselines

We provide a number of different baselines that learn from rewards. For RL baselines that leverage demonstrations see the [learning from demos section](../learning_from_demos/)

As part of these baselines we establish standardized [reinforcement learning benchmarks](#standard-benchmark) that cover a wide range of difficulties (easy to solve for verification but not saturated) and diversity in types of robotics task, including but not limited to classic control, dextrous manipulation, table-top manipulation, mobile manipulation etc.


## Online Reinforcement Learning Baselines

List of already implemented and tested online reinforcement learning baselines

| Baseline                                                            | Code                                                                           | Results | Paper                                    |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------- | ---------------------------------------- |
| Proximal Policy Optimization (PPO)                                  | [Link](https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/ppo) | WIP     | [Link](http://arxiv.org/abs/1707.06347)  |
| Soft Actor Critic (SAC)                                             | [Link](https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/sac) | WIP     | [Link](https://arxiv.org/abs/1801.01290) |
| Temporal Difference Learning for Model Predictive Control (TD-MPC2) | WIP                                                                            | WIP     | [Link](https://arxiv.org/abs/2310.16828) |

## Standard Benchmark

The standard benchmark for RL in ManiSkill consists of two groups, a small set of 10 tasks, and a large set of 50 tasks, both with state based and visual based settings. All standard benchmark tasks come with normalized dense reward functions and a currently unsolved tasks. The large set is still being developed and tested. 


These tasks span an extremely wide range of problems in robotics/reinforcement learning, namely: high dimensional observations/actions, large initial state distributions, locomotion, generalizable manipulation, mobile manipulation etc.


**Small Set Environment IDs**: 
PushCube-v1, PickCube-v1, StackCube-v1, PegInsertionSide-v1, PushT-v1, PickSingleYCB-v1, PlugCharger-v1, OpenCabinetDrawer-v1, HumanoidPlaceAppleInBowl-v1, AnymalC-Reach-v1
<!-- TODO: add image of all tasks / gif of them -->

<!-- 
**Large Set Environment IDs**: TODO 
add large collage image of all tasks
-->