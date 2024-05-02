# ManiSkill
ManiSkill is a powerful unified framework for robot simulation and training powered by [SAPIEN](https://sapien.ucsd.edu/). The entire stack is as open-source as possible. Among its features, it includes
- GPU parallelized visual data collection system. A policy can collect RGBD + Segmentation data at 20k FPS with a 4090 GPU, 10-100x faster compared to most other simulators.
- Example tasks covering a wide range of different robot embodiments (quadruped, mobile manipulators, single-arm robots) as well as a wide range of different tasks (table-top, locomotion, dextrous manipulation)
- GPU parallelized tasks, enabling incredibly fast synthetic data collection in simulation
- GPU parallelized tasks support simulating diverse scenes where every parallel environment has a completely different scene/set of objects
- Flexible task building API that abstracts away much of the complex GPU memory management code


## User Guide

A user guide on how to use ManiSkill with GPU parallelized simulation for your robotics and machine learning workflows
```{toctree}
:maxdepth: 2

user_guide/index
tasks/index
robots/index
contributing/index
roadmap/index
```