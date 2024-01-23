# Benchmarking

Code here is used to benchmark the performance of various simulators/benchmarks

To benchmark ManiSkill + SAPIEN, run

```
python benchmark_gpu_sim.py --num-envs=1024 --obs-mode=state # test just state simulation
python benchmark_gpu_sim.py --num-envs=128 --obs-mode=rgbd # test state sim + parallel rendering
python benchmark_gpu_sim.py --num-envs=128 --obs-mode=state --save-video # save a video showing all 128 visual observations
```