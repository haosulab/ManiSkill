# User Guide


```{figure} env_sample.png
---
alt: 4x4 grid of various usable tasks in ManiSkill
---
```


ManiSkill is a feature-rich GPU-accelerated robotics benchmark built on top of [SAPIEN](https://github.com/haosulab/sapien) designed to provide accessible support for a wide array of applications from robot learning, learning from demonstrations, sim2real/real2sim, and more. 

Features:

* GPU parallelized simulation enabling 250,000+ FPS on some tasks
* GPU parallelized rendering enabling 15,000+ FPS on some tasks, massively outperforming other simulators
* Flexible API to build custom tasks
* Variety of verified robotics tasks with diverse dynamics and visuals, from dexterous hands to low-level mobile manipulation
* Reproducible baselines in Reinforcement Learning and Learning from Demonstrations



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
<!-- additional_resources/education -->
```