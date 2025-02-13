# Performance Benchmarking

See [the performance benchmarking documentation](https://maniskill.readthedocs.io/en/latest/user_guide/additional_resources/performance_benchmarking.html) for in depth details.

If you plan to run the code here you need to git clone ManiSkill and change your directory to this one before running the code

Code Structure:
- `scripts/`: Bash scripts to run a matrix of performance tests. Results are saved to a local `benchmark_results` folder
- `plot_results.py`: Run this code to generate graphs of performance results saved to `benchmark_results`
- `envs/`: custom environments built for benchmarking, designed to be as close as possible between different simulators. Currently only Cartpole environment is tuned correctly for benchmarking across all simulators.


## Setup

### ManiSkill

See https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html and then run

```
pip install pynvml
```

<!-- ### Mujoco / MJX -->

<!-- ```bash
mamba create -n "mujoco_benchmarking" "python==3.11"
mamba activate mujoco_benchmarking
pip install mujoco-mjx
pip install "jax[cuda12]"
``` -->

### Isaac Lab

See https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html to create a conda/mamba environment.

Then run `pip install pynvml tyro pandas`.

## Running the Benchmark

All scripts are provided in the scripts folder that you can simply run directly. Otherwise example usages are shown below for benchmarking simulation and simulation+rendering FPS.

See the `scripts/` folder for the full list of commands used to generate official results, those commands save results to the `benchmark_results` folder in a .csv format. Running a benchmark with the same configurations of cameras/number of environments/choice of GPU will override the previous result. Example commands are shown below

### ManiSkill

```bash
python gpu_sim.py -e "CartpoleBalanceBenchmark-v1" \
    -n=2048 -o=state

python gpu_sim.py -e "CartpoleBalanceBenchmark-v1" \
    -n=1024 -o=rgb --num-cams=1 --cam-width=256 --cam-height=256

python gpu_sim.py -e "FrankaMoveBenchmark-v1" \
    -n=2048 -o=state --sim-freq=100 --control-freq=50

python gpu_sim.py -e "FrankaPickCubeBenchmark-v1" \
    -n=2048 -o=state --sim-freq=100 --control-freq=50
```

### Isaac Lab

```bash
python isaac_lab_gpu_sim.py --task Isaac-Cartpole-Direct-Benchmark-v0 --headless \
    --num-envs=2048 --obs-mode=state --save-results

python isaac_lab_gpu_sim.py --task Isaac-Cartpole-RGB-Camera-Direct-Benchmark-v0 --headless \
    --num-cams=1 --cam-width=512 --cam-height=512 --enable_cameras \
    --num-envs=128 --obs-mode=rgb --save-results
```

## Generating Plots

Comparing ManiSkill and Isaac Lab
```bash
python plot_results.py -e CartpoleBalanceBenchmark-v1 -f benchmark_results/maniskill.csv benchmark_results/isaac_lab.csv
```
