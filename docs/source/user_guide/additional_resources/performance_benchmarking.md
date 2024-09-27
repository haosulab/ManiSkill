# Performance Benchmarking

This page documents code and results of benchmarking various robotics simulators on a number of dimensions. It is still a WIP as we write more fair benchmarking code that more accurately compares simulators under the same conditions. We currently only have public graphs/results on one environment.

Currently we just compare ManiSkill and [IsaacLab](https://github.com/isaac-sim/IsaacLab) on one task, Cartpole Balancing (control). For details on benchmarking methodology see [this section](#benchmarking-detailsmethodology)

## Results

Raw benchmark results can be read from the .csv files in the [results folder on GitHub](https://github.com/haosulab/ManiSkill/blob/main/docs/source/user_guide/additional_resources/benchmarking_results). There are also plotted figures in that folder. Below we show a selection of some of the figures/results from testing on a RTX 4090. The figures are also sometimes annotated with the GPU memory usage in GB. 

Overall, ManiSkill is faster than IsaacLab on the majority of settings and is much more GPU memory efficient, especially for realistic camera setups. GPU memory efficiency is particularly important for machine learning methods like RL which rely on large replay buffers on the GPU. 

### Cartpole Balance

#### State

CartPoleBalance simulation only performance results showing FPS vs number of environments, annotated by GPU memory usage in GB on top of data points.
:::{figure} benchmarking_results/rtx_4090/fps:num_envs_state.png
:::

#### Realistic RGB Camera Setups

The [Open-X](https://robotics-transformer-x.github.io/) and [Droid](https://droid-dataset.github.io/) datasets are two of the largest real-world robotics datasets. Open-X typically has a single 640x480 RGB observation while Droid has 3 320x180 RGB observations. The next 2 figures show the performance of simulators when mimicing the real world camera setups.

:::{figure} benchmarking_results/rtx_4090/fps:rt_dataset_setup_bar.png
:::

:::{figure} benchmarking_results/rtx_4090/fps:droid_dataset_setup_bar.png
:::

#### RGB

CartPoleBalance simulation+rendering (rgb only) performance results showing FPS vs number of environments, annotated by GPU memory usage in GB on top of data points.
:::{figure} benchmarking_results/rtx_4090/fps:num_envs_1x512x512_rgb.png
:::

:::{figure} benchmarking_results/rtx_4090/fps:num_envs_1x256x256_rgb.png
:::

:::{figure} benchmarking_results/rtx_4090/fps:num_envs_1x128x128_rgb.png
:::

#### RGB+Depth

CartPoleBalance simulation+rendering (rgb+depth) performance results showing FPS vs number of environments, annotated by GPU memory usage in GB on top of data points.
:::{figure} benchmarking_results/rtx_4090/fps:num_envs_1x512x512_rgb+depth.png
:::

:::{figure} benchmarking_results/rtx_4090/fps:num_envs_1x256x256_rgb+depth.png
:::

:::{figure} benchmarking_results/rtx_4090/fps:num_envs_1x128x128_rgb+depth.png
:::
## Commands for Reproducing the Results

See the scripts under [mani_skill/examples/benchmarking/scripts](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/examples/benchmarking/scripts)

## Benchmarking Details/Methodology

There is currently one benchmarked task: Cartpole Balance (classic control). Details about the exact configurations of the two tasks are detailed in the [next section](#task-configuration).

For simulators that use physx (like ManiSkill and Isaac Lab), for comparison we try to align as many simulation configuration parameters (like number of solver position iterations) as well as object types (e.g. collision meshes, size of objects etc.) as close as possible.

In the future we plan to benchmark other simulators using other physics engines (like Mujoco) although how to fairly do so is still WIP.

Reward functions and evaluation functions are purposely left out and not benchmarked.

Due to varying implementations of parallel rendering mechanisms, to benchmark the FPS when generating RGB, Depth, and/or Segmentation observations, we plot figures showing the FPS ablating on the number of parallel environments, image size, and number of cameras. Within each evaluation, we try and pick configurations that maximize the FPS when given fixed # of environments, # of cameras per environment, and/or image size.

## Task Configuration

WIP