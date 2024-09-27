# ManiSkill 3 (Beta)


![teaser](figures/teaser.jpg)
<p style="text-align: center; font-size: 0.8rem; color: #999;margin-top: -1rem;">Sample of environments/robots rendered with ray-tracing. Scene datasets sourced from AI2THOR and ReplicaCAD</p>

[![Downloads](https://static.pepy.tech/badge/mani_skill)](https://pepy.tech/project/mani_skill)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haosulab/ManiSkill/blob/main/examples/tutorials/1_quickstart.ipynb)
[![PyPI version](https://badge.fury.io/py/mani-skill.svg)](https://badge.fury.io/py/mani-skill)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://maniskill.readthedocs.io/en/latest/)
[![Discord](https://img.shields.io/discord/996566046414753822?logo=discord)](https://discord.gg/x8yUZe5AdN)

ManiSkill is a powerful unified framework for robot simulation and training powered by [SAPIEN](https://sapien.ucsd.edu/). The entire stack is as open-source as possible and ManiSkill v3 is in beta release now. Among its features include:
- GPU parallelized visual data collection system. On the high end you can collect RGBD + Segmentation data at 20k FPS with a 4090 GPU, 10-100x faster compared to most other simulators.
- Example tasks covering a wide range of different robot embodiments (quadruped, mobile manipulators, single-arm robots) as well as a wide range of different tasks (table-top, locomotion, dextrous manipulation)
- GPU parallelized tasks, enabling incredibly fast synthetic data collection in simulation
- GPU parallelized tasks support simulating diverse scenes where every parallel environment has a completely different scene/set of objects
- Flexible task building API that abstracts away much of the complex GPU memory management code
<!-- - Evaluate models trained on real-world data in simulation, no robot hardware needed -->

ManiSkill plans to enable all kinds of workflows, including but not limited to 2D/3D vision-based reinforcement learning, imitation learning, sense-plan-act, etc. There will also be more assets/scenes supported by ManiSkill (e.g. [AI2THOR](https://ai2thor.allenai.org/)) in addition to other features such as [digital-twins for evaluation of real-world policies](https://github.com/simpler-env/SimplerEnv), see [our roadmap](https://maniskill.readthedocs.io/en/latest/roadmap/index.html) for planned features that will be added over time before the official v3 is released.

Please refer to our [documentation](https://maniskill.readthedocs.io/en/latest/user_guide) to learn more information from tutorials on building tasks to data collection.

**NOTE:**
This project currently is in a **beta release**, so not all features have been added in yet and there may be some bugs. If you find any bugs or have any feature requests please post them to our [GitHub issues](https://github.com/haosulab/ManiSkill/issues/) or discuss about them on [GitHub discussions](https://github.com/haosulab/ManiSkill/discussions/). We also have a [Discord Server](https://discord.gg/x8yUZe5AdN) through which we make announcements and discuss about ManiSkill.

Users looking for the original ManiSkill2 can find the commit for that codebase at the [v0.5.3 tag](https://github.com/haosulab/ManiSkill/tree/v0.5.3)


## Installation
Installation of ManiSkill is extremely simple, you only need to run a few pip installs

```bash
# install the package
pip install --upgrade mani_skill
# install a version of torch that is compatible with your system
pip install torch torchvision torchaudio
```

Finally you also need to set up Vulkan with [instructions here](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan)

For more details about installation (e.g. from source, or doing troubleshooting) see [the documentation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html
)

## Getting Started

To get started, check out the quick start documentation: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html

We also have a quick start [colab notebook](https://colab.research.google.com/github/haosulab/ManiSkill/blob/main/examples/tutorials/1_quickstart.ipynb) that lets you try out GPU parallelized simulation without needing your own hardware. Everything is runnable on Colab free tier.

For a full list of example scripts you can run see [the docs](https://maniskill.readthedocs.io/en/latest/user_guide/demos/index.html).

## System Support

We currently best support Linux based systems. There is limited support for windows and no support for MacOS at the moment. We are working on trying to support more features on other systems but this may take some time. Most constraints stem from what the [SAPIEN](https://github.com/haosulab/SAPIEN/) package is capable of supporting.

| System / GPU         | CPU Sim | GPU Sim | Rendering |
| -------------------- | ------- | ------- | --------- |
| Linux / NVIDIA GPU   | ✅      | ✅      | ✅        |
| Windows / NVIDIA GPU | ✅      | ❌      | ✅        |
| Windows / AMD GPU    | ✅      | ❌      | ✅        |
| WSL / Anything       | ✅      | ❌      | ❌        |
| MacOS / Anything     | ❌      | ❌      | ❌        |

## Citation and Core Team

ManiSkill 3 will have a technical paper coming out. 

The current author list is as follows: Stone Tao*, Fanbo Xiang*, Arth Shukla, Chen Bao, Nan Xiao, Rui Chen, Tongzhou Mu, Tse-Kai Chan, Xander Hinrichsen, Xiaodi Yuan, Xinsong Lin, Xuanlin Li, Yuan Gao, Yuzhe Qin, Zhiao Huang, Hao Su

## License

All rigid body environments in ManiSkill are licensed under fully permissive licenses (e.g., Apache-2.0).

The assets are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode).
