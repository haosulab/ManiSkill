# Baselines

We provide a number of different baselines spanning different categories of learning from demonstrations research: Behavior Cloning / Supervised Learning, Offline Reinforcement Learning, and Online Learning from Demonstrations. This page is still a WIP as we finish running experiments and establish clear baselines and benchmarking setups.

<!-- As part of these baselines we establish a few standard learning from demonstration benchmarks that cover a wide range of difficulty (easy to solve for verification but not saturated) and diversity in types of demonstrations (human collected, motion planning collected, neural net policy generated) -->

**Behavior Cloning (BC) Baselines**

BC Baselines are characterized by supervised learning focused algorithms for learning from demonstrations, without any online interaction with the environment.

| Baseline                       | Code                                                                                        | Results | Paper                                      |
| ------------------------------ | ------------------------------------------------------------------------------------------- | ------- | ------------------------------------------ |
| Standard Behavior Cloning (BC) | WIP                                                                                         | WIP     | N/A                                        |
| Diffusion Policy (DP)          | [Link](https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/diffusion_policy) | WIP     | [Link](https://arxiv.org/abs/2303.04137v4) |
| Action Chunking Transformer (ACT) | [Link](https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/act) | WIP     | [Link](https://arxiv.org/abs/2304.13705) |

**Online Learning from Demonstrations Baselines**

Online learning from demonstrations baselines are characterized by learning from demonstrations while also leveraging online environment interactions. 

| Baseline                                      | Code                                                                            | Results | Paper                                    |
| --------------------------------------------- | ------------------------------------------------------------------------------- | ------- | ---------------------------------------- |
| Reverse Forward Curriculum Learning (RFCL)*   | [Link](https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/rfcl) | WIP     | [Link](https://arxiv.org/abs/2405.03379) |
| Reinforcement Learning from Prior Data (RLPD) | [Link](https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/rlpd) | WIP     | [Link](https://arxiv.org/abs/2302.02948) |
| SAC + Demos (SAC+Demos)                       | WIP                                                                             | N/A     |                                          |


\* - This indicates the baseline uses environment state reset which is typically a simulation only feature 