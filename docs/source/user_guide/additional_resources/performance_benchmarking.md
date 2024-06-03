# Performance Benchmarking

This page documents code and results of benchmarking various robotics simulators on a number of dimensions. It is still a WIP as we write more fair benchmarking code that more accurately compares simulators under the same conditions.

## Benchmarking Details/Methodology
WIP


## ManiSkill

To benchmark ManiSkill + SAPIEN, after following the setup instructions on this repository's README.md, run

```
python -m mani_skill.examples.benchmarking.gpu_sim -e "PickCube-v1" -n=4096 -o=state --control-freq=50
python -m mani_skill.examples.benchmarking.gpu_sim -e "PickCube-v1" -n=1536 -o=rgbd --control-freq=50
# note we use --control-freq=50 as this is the control frequency isaac sim based repos tend to use
```

These are the expected state-based only results on a single 4090 GPU:
```
env.step: 277840.711 steps/s, 67.832 parallel steps/s, 100 steps in 1.474s
env.step+env.reset: 239463.964 steps/s, 58.463 parallel steps/s, 1000 steps in 17.105s
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
