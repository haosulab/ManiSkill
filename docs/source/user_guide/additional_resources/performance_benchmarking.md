# Performance Benchmarking

This page documents code and results of benchmarking various robotics simulators on a number of dimensions. It is still a WIP as we write more fair benchmarking code that more accurately compares simulators under the same conditions.

## Benchmarking Details/Methodology

There are currently two benchmarked tasks: Cartpole Balance (classic control), and Pick Cube (manipulation). Details about the exact configurations of the two tasks are detailed in [this section](#task-configuration).

For simulators that use physx (like ManiSkill and Isaac Lab), for comparison we try to align as many simulation configuration parameters (like number of solver position iterations) as well as object types (e.g. collision meshes, size of objects etc.) as close as possible.

In the future we plan to benchmark other simulators using other physics engines (like Mujoco) although how to fairly do so is still WIP.

Reward functions and evaluation functions are purposely left out and not benchmarked.

Due to varying implementations of parallel rendering mechanisms, to benchmark the FPS when generating RGB, Depth, and/or Segmentation observations, we plot figures showing the FPS ablating on the number of parallel environments, image size, and number of cameras. Within each evaluation, we try and pick configurations that maximize the FPS when given fixed # of environments, # of cameras per environment, and image size.

## Results

Results below are what occur when following the methodology above. The tables all show a snapshot of results using 4096 environments for measuring state-based FPS and the max number of environments runnable on a 4090 GPU using one 128x128 camera per environment. 

### Cartpole Balance

| Simulator/Framework | State FPS   | RGB FPS   | Depth FPS | RGB+Depth FPS | RGB+Depth+Segmentation FPS |
| ------------------- | ----------- | --------- | --------- | ------------- | -------------------------- |
| SAPIEN/ManiSkill    | 2537724.820 | 33018.081 | 33018.081 | 33018.081     | 33018.081                  |
| IsaacSim/IsaacLab   |             |           |           | N/A           | N/A                        |

### Pick Cube

| Simulator/Framework | State FPS  | RGB FPS   | Depth FPS | RGB+Depth FPS | RGB+Depth+Segmentation FPS |
| ------------------- | ---------- | --------- | --------- | ------------- | -------------------------- |
| SAPIEN/ManiSkill    | 318580.257 | 18549.002 | 18549.002 | 18549.002     | 18549.002                  |
| IsaacSim/IsaacLab   |            |           |           | N/A           | N/A                        |

## Commands for Reproducing the Results

### ManiSkill

To benchmark SAPIEN / ManiSkill, after following the setup instructions on this repository's README.md, run

```
# test state simulation
python -m mani_skill.examples.benchmarking.gpu_sim -e "PickCubeBenchmark-v1" -n=4096 -o=state
# test rendering RGB, Depth, and Segmentation
python -m mani_skill.examples.benchmarking.gpu_sim -e "PickCubeBenchmark-v1" -n=1024 -o=rgbd
```
The environments available are "PickCubeBenchmark-v1" and "CartpoleBalanceBenchmark-v1" (modified versions of the default environment in ManiSkill for benchmarking purposes)

To generate the plots you can run

## Isaac Lab

To benchmark [Isaac Sim / Isaac Lab](https://github.com/isaac-sim/IsaacLab), follow their installation instructions here https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html. We recommend making a conda/mamba environment to install it. Then after activating the environment, run

```
cd mani_skill/examples/benchmarking
# test state simulation
python isaac_lab_gpu_sim.py --task "Isaac-Cartpole-Direct-v0" --num_envs 4096  --headless
# test rendering just RGB
python isaac_lab_gpu_sim.py --task "Isaac-Cartpole-RGB-Camera-Direct-Benchmark-v0" \
  --num_envs 256 --enable_cameras --headless
# test rendering RGB+Depth
python isaac_lab_gpu_sim.py --task "Isaac-Cartpole-RGB-Camera-Direct-Benchmark-v0" \
  --num_envs 4 --obs_mode rgbd \
  --enable_cameras --headless
```


## Task Configuration