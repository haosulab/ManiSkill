# Sim2Real Tools

ManiSkill provides some useful tooling for doing sim2real training and deployment. This document discusses the `Sim2RealEnv` class and how it can help you perform sim2real experiments with less bugs and better sim2real alignment. It is intended for users who wish to do more advanced sim2real and/or build their own simulation environments to prepare for sim2real work.

We also have a simple tutorial that shows how to use visual Reinforcement Learning to train a cube picking policy in simulation using the low-costSO100 robot arm and [🤗 LeRobot](https://github.com/huggingface/lerobot) system. At the end of the tutorial you will be able to deploy zero-shot a RGB policy successfully without needing any real world data (although it can help!). As there a lot of code that is often very specific to certain hardware and sim2real approaches, the tutorial an code for this particular sim2real approach is at our [LeRobot-Sim2Real repository](https://github.com/StoneT2000/lerobot-sim2real/). The LeRobot-Sim2Real tutorial does not require you to follow any of the other tutorials and is self-contained.

Linked below are tutorials/docmentation on some tools and how to build with them. The Sim2RealEnv details how to align your simulated environment with the real world better using just a single wrapper. The Building Simulation Environments for Sim2Real outlines steps and details to follow to ensure the simulation environment is ready for sim2real (currently WIP).

```{toctree}
:titlesonly:
sim2realenv
building_sim2real_envs
```