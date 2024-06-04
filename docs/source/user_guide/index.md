# User Guide


```{figure} teaser.png
```
<p style="text-align: center; font-size: 0.8rem; color: #999;margin-top: -1rem;">Sample of environments/robots rendered with ray-tracing. Scene datasets sourced from AI2THOR and ReplicaCAD</p>

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

**NOTE:**
This project currently is in a **beta release**, so not all features have been added in yet and there may be some bugs. If you find any bugs or have any feature requests please post them to our [GitHub issues](https://github.com/haosulab/ManiSkill/issues/) or discuss about them on [GitHub discussions](https://github.com/haosulab/ManiSkill/discussions/). We also have a [Discord Server](https://discord.gg/x8yUZe5AdN) through which we make announcements and discuss about ManiSkill.

```{toctree}
:caption: Get started

getting_started/installation
getting_started/quickstart
getting_started/docker
```

```{toctree}
:maxdepth: 3
:caption: Resources

demos/index
tutorials/index
concepts/index
datasets/index
data_collection/index
```

```{toctree}
:maxdepth: 2
:caption: Additional Resources

additional_resources/performance_benchmarking
reference/index.rst
```