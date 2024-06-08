# Learning from Demonstrations / Imitation Learning

We provide a number of different baselines spanning different categories of learning from demonstrations research: Behavior Cloning / Supervised Learning, Offline Reinforcement Learning, and Online Learning from Demonstrations.

As part of these baselines we establish a few standard learning from demonstration benchmarks that cover a wide range of difficulty (easy to solve for verification but not saturated) and diversity in types of demonstrations (human collected, motion planning collected, neural net policy generated)

**Behavior Cloning Baselines**
| Baseline                           | Code | Results |
| ---------------------------------- | ---- | ------- |
| Standard Behavior Cloning (BC) | WIP  | WIP     |
| Diffusion Policy (DP)                   | WIP  | WIP     |
| Action Chunk Transformers (ACT)    | WIP  | WIP     |


**Online Learning from Demonstrations Baselines**

| Baseline                                            | Code                                                                                | Results | Paper                                    |
| --------------------------------------------------- | ----------------------------------------------------------------------------------- | ------- | ---------------------------------------- |
| Reverse Forward Curriculum Learning (RFCL)* | [Link](https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/rfcl) | WIP     | [Link](https://arxiv.org/abs/2405.03379) |
| Reinforcement Learning from Prior Data (RLPD)       | WIP                                                                                 | WIP     | [Link](https://arxiv.org/abs/2302.02948) |
| SAC + Demos (SAC+Demos)                             | WIP                                                                                 | N/A     |                                          |


\* - This indicates the baseline uses environment state reset which is typically a simulation only feature 