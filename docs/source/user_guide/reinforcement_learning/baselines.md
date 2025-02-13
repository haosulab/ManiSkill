# Baselines

We provide a number of different baselines that learn from rewards via online reinforcement learning.
<!-- For RL baselines that leverage demonstrations see the [learning from demos section](../learning_from_demos/) -->

As part of these baselines we establish standardized [reinforcement learning benchmarks](#standard-benchmark) that cover a wide range of difficulties (easy to solve for verification but not saturated) and diversity in types of robotics task, including but not limited to classic control, dextrous manipulation, table-top manipulation, mobile manipulation etc.


## Online Reinforcement Learning Baselines

List of already implemented and tested online reinforcement learning baselines. The results link takes you to the respective wandb pages for the results. You can change filters/views in the wandb workspace to view results with other settings (e.g. state based or RGB based training). Note that there are also reinforcement learning (offline RL, online imitation learning) baselines that leverage demonstrations, see the [learning from demos page](../learning_from_demos/index.md) for more information.

| Baseline                                                            | Code                                                                           | Results | Paper                                    |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------- | ---------------------------------------- |
| Proximal Policy Optimization (PPO)                                  | [Link](https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/ppo) | [Link](https://api.wandb.ai/links/stonet2000/k6lz966q)     | [Link](http://arxiv.org/abs/1707.06347)  |
| Soft Actor Critic (SAC)                                             | [Link](https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/sac) | WIP     | [Link](https://arxiv.org/abs/1801.01290) |
| Temporal Difference Learning for Model Predictive Control (TD-MPC2) | WIP                                                                            | WIP     | [Link](https://arxiv.org/abs/2310.16828) |

## Standard Benchmark

The standard benchmark for RL in ManiSkill consists of two groups, a small set of 8 tasks, and a large set of 50 tasks, both with state based and visual based settings. All standard benchmark tasks come with normalized dense reward functions. A recommended small set is created so researchers without access to a lot of compute can still reasonably benchmark/compare their work. The large set is still being developed and tested. 


These tasks span an extremely wide range of problems in robotics/reinforcement learning, namely: high dimensional observations/actions, large initial state distributions, articulated object manipulation, generalizable manipulation, mobile manipulation, locomotion etc.


**Small Set Environment IDs**: 
<!-- PushCube-v1, PickCube-v1, StackCube-v1, PegInsertionSide-v1, PushT-v1, PickSingleYCB-v1, PlugCharger-v1, OpenCabinetDrawer-v1, HumanoidPlaceAppleInBowl-v1, AnymalC-Reach-v1 -->
PushCube-v1, PickCube-v1, PegInsertionSide-v1, PushT-v1, HumanoidPlaceAppleInBowl-v1, AnymalC-Reach-v1, OpenCabinetDrawer-v1
<!-- TODO: add image of all tasks / gif of them -->

<!-- 
**Large Set Environment IDs**: TODO 
add large collage image of all tasks
-->
<!-- - TableTop: PushCube-v1, PickCube-v1, StackCube-v1, PegInsertionSide-v1, PushT-v1, PickSingleYCB-v1, PlugCharger-v1, RollBall-v1, PlaceSphere-v1, PullCube-v1, LiftPegUpRight-v1, TwoRobotPickCube-v1, TwoRobotStackCube-v1
- Mobile Manipulation: OpenCabinetDrawer-v1
- Humanoid: HumanoidPlaceAppleInBowl-v1
- Quadruped: AnymalC-Reach-v1, AnymalC-Spin-v1,
- Classic Control: MS-CartpoleBalance-v1, MS-CartpoleSwingup-v1, MS-HopperStand-v1, MS-HopperHop-v1 -->


## Evaluation

For proper evaluation of RL policies, see how that code is setup in the [evaluation section in the RL setup page](./setup.md#evaluation). All results reported in the results linked above follow the same evaluation setup.