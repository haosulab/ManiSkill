# Benchmarking (WIP)

Code here is used to benchmark the performance of various simulators/benchmarks under as fair conditions as possible. To keep tests fair, the code here ensures the simulators are simulating the same task, same robot and dynamic objects, and use the same simulation and control frequencies. 

This is still WIP as we work on more fair evaluations that align environments more closely and compare robot learning workflows.


## ManiSkill

To benchmark ManiSkill + SAPIEN, after following the setup instructions on this repository's README.md, run

```
python gpu_sim.py -e "PickCube-v1" --num-envs=1024 --obs-mode=state # test just state simulation
python gpu_sim.py -e "PickCube-v1" --num-envs=128 --obs-mode=rgbd # test state sim + parallel rendering one 128x128 RGBD cameras per environment
python gpu_sim.py -e "PickCube-v1" --num-envs=128 --save-video # save a video showing all 128 visual observations
```


To get the reported results, we run two commands on a machine with a RTX 4090:
```
python gpu_sim.py -e "PickCube-v1" --num-envs=4096 --obs-mode=state --control-freq=50
python gpu_sim.py -e "PickCube-v1" --num-envs=1536 --obs-mode=rgbd --control-freq=50
# note we use --control-freq=50 as this is the control frequency isaac sim based repos tend to use
```

These are the expected state-based only results:
```
env.step: 277248.895 steps/s, 67.688 parallel steps/s, 100 steps in 1.477s
env.step+env.reset: 252232.109 steps/s, 61.580 parallel steps/s, 1000 steps in 16.239s
```

These are the expected visual observations/rendering results:
```
env.step: 18549.002 steps/s, 12.076 parallel steps/s, 100 steps in 8.281s
env.step+env.reset: 17953.079 steps/s, 11.688 parallel steps/s, 1000 steps in 85.556s
```


## Isaac Lab

To benchmark [Isaac Lab](https://github.com/isaac-sim/IsaacLab), follow their installation instructions here https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html. We recommend making a conda/mamba environment to install it. Then after activating the environment, run

```
python isaac_lab_gpu_sim.py --task "Isaac-Lift-Cube-Franka-v0" --num_envs 4096 --headless
```


## Benchmarking Details/Methodology

See [the performance benchmarking documentation](https://maniskill.readthedocs.io/en/latest/user_guide/additional_resources/performance_benchmarking.html) for in depth details.