# Performance Benchmarking

This page documents code and results of benchmarking various robotics simulators on a number of dimensions. It is still a WIP as we write more fair benchmarking code that more accurately compares simulators under the same conditions. We currently only have public graphs/results on one environment.

Currently we just compare ManiSkill and [IsaacLab](https://github.com/isaac-sim/IsaacLab) on two tasks, Cartpole Balancing (control) and PickCube (table top manipulation). For details on benchmarking methodology see [this section](#benchmarking-detailsmethodology)

## Results

Raw benchmark results can be read from the .csv files in the [results folder on GitHub](https://github.com/haosulab/ManiSkill/blob/main/docs/source/user_guide/additional_resources/benchmarking_results). There are also plotted figures in that folder. Below we show a selection of some of the figures/results from testing on a RTX 3080. The figures are also sometimes annotated with the GPU memory usage in GB or the number of parallel environments used for that result.

*Note IsaacLab currently does not support RGB+Depth, or multiple cameras per sub-scene so there may not be results for IsaacLab on some figures

### Cartpole Balance

#### State

:::{figure} benchmarking_results/rtx_3080/fps:num_envs_state.png
:::

#### RGB

:::{figure} benchmarking_results/rtx_3080/fps:num_envs_1x256x256_rgb.png
:::

:::{figure} benchmarking_results/rtx_3080/fps:num_cameras_rgb.png
:::

:::{figure} benchmarking_results/rtx_3080/fps:camera_size_rgb.png
:::

#### RGB+Depth

ManiSkill renders RGB and RGB+Depth at the same speed, see figures above for results. IsaacLab currently does not support this.


## Commands for Reproducing the Results

See the scripts under [mani_skill/examples/benchmarking/scripts](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/examples/benchmarking/scripts)

## Benchmarking Details/Methodology

There are currently two benchmarked tasks: Cartpole Balance (classic control), and Pick Cube (manipulation). Details about the exact configurations of the two tasks are detailed in the [next section](#task-configuration).

For simulators that use physx (like ManiSkill and Isaac Lab), for comparison we try to align as many simulation configuration parameters (like number of solver position iterations) as well as object types (e.g. collision meshes, size of objects etc.) as close as possible.

In the future we plan to benchmark other simulators using other physics engines (like Mujoco) although how to fairly do so is still WIP.

Reward functions and evaluation functions are purposely left out and not benchmarked.

Due to varying implementations of parallel rendering mechanisms, to benchmark the FPS when generating RGB, Depth, and/or Segmentation observations, we plot figures showing the FPS ablating on the number of parallel environments, image size, and number of cameras. Within each evaluation, we try and pick configurations that maximize the FPS when given fixed # of environments, # of cameras per environment, and/or image size.

## Task Configuration
WIP