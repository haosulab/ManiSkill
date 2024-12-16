# User Guide

![teaser](./teaser.jpg)
<p style="text-align: center; font-size: 0.8rem; color: #999;margin-top: -1rem;">Sample of environments/robots rendered with ray-tracing. Scene datasets sourced from AI2THOR and ReplicaCAD</p>

[![Downloads](https://static.pepy.tech/badge/mani_skill)](https://pepy.tech/project/mani_skill)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haosulab/ManiSkill/blob/main/examples/tutorials/1_quickstart.ipynb)
[![PyPI version](https://badge.fury.io/py/mani-skill.svg)](https://badge.fury.io/py/mani-skill)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://maniskill.readthedocs.io/en/latest/)
[![Discord](https://img.shields.io/discord/996566046414753822?logo=discord)](https://discord.gg/x8yUZe5AdN)

ManiSkill is a powerful unified framework for robot simulation and training powered by [SAPIEN](https://sapien.ucsd.edu/), with a strong focus on manipulation skills. The entire tech stack is as open-source as possible and ManiSkill v3 is in beta release now. Among its features include:
- GPU parallelized visual data collection system. On the high end you can collect RGBD + Segmentation data at 30,000+ FPS with a 4090 GPU, 10-1000x faster compared to most other simulators.
- GPU parallelized simulation, enabling high throughput state-based synthetic data collection in simulation
- GPU parallelized heteogeneous simuluation, where every parallel environment has a completely different scene/set of objects
- Example tasks cover a wide range of different robot embodiments (humanoids, mobile manipulators, single-arm robots) as well as a wide range of different tasks (table-top, drawing/cleaning, dextrous manipulation)
- Flexible and simple task building API that abstracts away much of the complex GPU memory management code via an object oriented design
- Real2sim environments for scalably evaluating real-world policies 60-100x faster via GPU simulation.

<!-- TODO replace paper link with arxiv link when it is out -->
For more details we encourage you to take a look at our [paper](https://github.com/haosulab/ManiSkill/blob/main/figures/maniskill3_paper.pdf).

There are more features to be added to ManiSkill 3, see [our roadmap](https://maniskill.readthedocs.io/en/latest/roadmap/index.html) for planned features that will be added over time before the official v3 is released.

Please refer to our [documentation](https://maniskill.readthedocs.io/en/latest/user_guide) to learn more information from tutorials on building tasks to data collection.

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
reinforcement_learning/index
learning_from_demos/index
wrappers/index
```

```{toctree}
:maxdepth: 2
:caption: Additional Resources

additional_resources/performance_benchmarking
additional_resources/citation
reference/index.rst
```