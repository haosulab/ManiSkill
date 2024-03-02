# Benchmarking

Code here is used to benchmark the performance of various simulators/benchmarks under as fair conditions as possible. To keep tests fair, the code here ensures the simulators are simulating the same task, same robot and dynamic objects, and use the same simulation and control frequencies. 


## ManiSkill

To benchmark ManiSkill + SAPIEN, after following the setup instructions on this repository's README.md, run

```
python benchmark_maniskill.py -e "PickCube-v1" --num-envs=1024 --obs-mode=state # test just state simulation
python benchmark_maniskill.py -e "PickCube-v1" --num-envs=128 --obs-mode=rgbd # test state sim + parallel rendering one 128x128 RGBD cameras per environment
python benchmark_maniskill.py -e "PickCube-v1" --num-envs=128 --save-video # save a video showing all 128 visual observations
```


To get the reported results, we run two commands on a machine with a RTX 4090:
```
python benchmark_maniskill.py -e "PickCube-v1" --num-envs=4096 --obs-mode=state --control-freq=50
python benchmark_maniskill.py -e "PickCube-v1" --num-envs=1536 --obs-mode=rgbd --control-freq=50
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


## Isaac Orbit

To benchmark [Isaac Orbit](https://github.com/NVIDIA-Omniverse/orbit), follow their installation instructions here https://isaac-orbit.github.io/orbit/source/setup/installation.html. We recommend making a conda/mamba environment to install it. Then after activating the environment, run

```
python benchmark_orbit_sim.py --task "Isaac-Lift-Cube-Franka-v0" --num_envs 4096 --headless
```

These are the expected results as tested on a RTX 4090:
```
env.step: 41043.659 steps/s, 10.020 parallel steps/s, 100 steps in 9.980s
env.step+env.reset: 53911.399 steps/s, 13.162 parallel steps/s, 1000 steps in 75.977s
```

Note that the FPS appears to be much lower than reported by Isaac Orbit's original paper (up to 125,000 FPS). It is unclear why that is at the moment but even then as it stands ManiSkill tasks are much faster by at least 2-3x.

## OmniIsaacGymEnvs

To benchmark [OmniIsaacGymEnvs](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs), follow their installation instructions on their repository 

```
alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-*/python.sh
PYTHON_PATH scripts/benchmark_omniisaac_sim.py headless=True num_envs=4096 task="FrankaCabinet"
```
<!-- Notes: FrankaCabinet uses solver iterations 12 and 120:60 sim to control freq, which makes it run faster than other sims -->
<!-- 
## RoboSuite

To benchmark [Robosuite](https://github.com/ARISE-Initiative/robosuite) (powered by Mujoco), run

Note that RoboSuite currently has not integrated the new MJX which brings GPU parallelized simulation so speeds here are expectedly much lower. Once MJX is supported we will benchmark their GPU simulation properly. -->

