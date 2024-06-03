# Performance Benchmarking

This page documents code and results of benchmarking various robotics simulators on a number of dimensions. It is still a WIP as we write more fair benchmarking code that more accurately compares simulators under the same conditions.

## Benchmarking Details/Methodology

There are currently two benchmarked tasks: Cartpole Balance (classic control), and Pick Cube (manipulation). Details about the exact configurations of the two tasks are detailed in [this section](#task-configuration).

For simulators that use physx (like ManiSkill and Isaac Lab), for comparison we try to align as many simulation configuration parameters (like number of solver position iterations) as well as object types (e.g. collision meshes, size of objects etc.) as close as possible.

In the future we plan to benchmark other simulators using other physics engines (like Mujoco) although how to fairly do so is still WIP.

Reward functions and evaluation functions are purposely left out and not benchmarked.

Due to varying implementations of parallel rendering mechanisms, to benchmark the FPS when generating RGB, Depth, and/or Segmentation observations, we plot figures showing the FPS ablating on the number of parallel environments, image size, and number of cameras. Within each evaluation, we try and pick configurations that maximize the FPS when given fixed # of environments, # of cameras per environment, and image size.

## Results

Results below are what occur when following the methodology above. The tables all show a snapshot of results using 4096 environments for measuring state-based FPS and the max number of environments runnable on a 4090 GPU using 128x128 camera size 

### Cartpole Balance

| Simulator/Framework | State FPS | RGB FPS | Depth FPS | RGB+Depth Observation FPS | RGB+Depth+Segmentation Observation FPS |
| ------------------- | --------- | ------- | --------- | ------------------------- | -------------------------------------- |
| SAPIEN/ManiSkill    |           |         |           |                           |                                        |
| IsaacSim/IsaacLab   |           |         |           | N/A                       | N/A                                    |

### Pick Cube

| Simulator/Framework | State FPS  | RGB FPS   | Depth FPS | RGB+Depth Observation FPS | RGB+Depth+Segmentation Observation FPS |
| ------------------- | ---------- | --------- | --------- | ------------------------- | -------------------------------------- |
| SAPIEN/ManiSkill    | 318580.257 | 18549.002 | 18549.002 | 18549.002                 | 18549.002                              |
| IsaacSim/IsaacLab   |            |           |           | N/A                       | N/A                                    |

## Commands for Reproducing the Results

## ManiSkill

To benchmark ManiSkill + SAPIEN, after following the setup instructions on this repository's README.md, run

```
python -m mani_skill.examples.benchmarking.gpu_sim -e "PickCubeBenchmark-v1" -n=4096 -o=state
python -m mani_skill.examples.benchmarking.gpu_sim -e "PickCubeBenchmark-v1" -n=1536 -o=rgbd
```
The environments available are "PickCubeBenchmark-v1" and "CartpoleBalanceBenchmark-v1" (modified versions of the default environment in ManiSkill for benchmarking purposes)


These are the expected state-based only results on a single 4090 GPU:
```
env.step: 318580.257 steps/s, 77.778 parallel steps/s, 1000 steps in 12.857s
env.step+env.reset: 316447.114 steps/s, 77.258 parallel steps/s, 1000 steps in 12.944s
```

These are the expected visual observations/rendering results on a single 4090 GPU:
```
env.step: 18549.002 steps/s, 12.076 parallel steps/s, 100 steps in 8.281s
env.step+env.reset: 18146.848 steps/s, 11.814 parallel steps/s, 1000 steps in 84.643s
```


## Isaac Lab

To benchmark [Isaac Lab](https://github.com/isaac-sim/IsaacLab), follow their installation instructions here https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html. We recommend making a conda/mamba environment to install it. Then after activating the environment, run

```
cd mani_skill/examples/benchmarking
# test state simulation
python isaac_lab_gpu_sim.py --task "Isaac-Lift-Cube-Franka-v0" --num_envs 4096 --headless
# test rendering just RGB
python isaac_lab_gpu_sim.py --task "Isaac-Cartpole-RGB-Camera-Direct-v0" --num_envs 256 --enable_cameras --headless
```


## Task Configuration