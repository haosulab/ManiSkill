# User Guide


```{figure} env_sample.png
---
alt: 4x4 grid of various usable tasks in ManiSkill
---
```


ManiSkill is a powerful unified framework for robot simulation and training powered by [SAPIEN](https://sapien.ucsd.edu/). The entire stack is as open-source as possible. Among its features, it includes
- GPU parallelized visual data collection system. A policy can collect RGBD + Segmentation data at 20k FPS with a 4090 GPU, 10-100x faster compared to most other simulators.
- Example tasks covering a wide range of different robot embodiments (quadruped, mobile manipulators, single-arm robots) as well as a wide range of different tasks (table-top, locomotion, dextrous manipulation)
- GPU parallelized tasks, enabling incredibly fast synthetic data collection in simulation
- GPU parallelized tasks support simulating diverse scenes where every parallel environment has a completely different scene/set of objects
- Flexible task building API that abstracts away much of the complex GPU memory management code



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
<!-- algorithms_and_models/index
workflows/index -->


```{toctree}
:maxdepth: 2
:caption: Additional Resources

additional_resources/performance_benchmarking
reference/index.rst
```