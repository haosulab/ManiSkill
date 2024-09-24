# Performance Benchmarking

See [the performance benchmarking documentation](https://maniskill.readthedocs.io/en/latest/user_guide/additional_resources/performance_benchmarking.html) for in depth details.

If you plan to run the code here you need to git clone ManiSkill and change your directory to this one before running the code

Code Structure:
- `scripts/`: Bash scripts to run a matrix of performance tests. Results are saved to a local `benchmark_results` folder
- `plot_results.py`: Run this code to generate graphs of performance results saved to `benchmark_results`
- `envs/`: custom environments built for benchmarking, designed to be as close as possible between different simulators. Currently only Cartpole environment is tuned correctly for benchmarking.


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

Then run `pip install pynvml` to install pynvml which is used for GPU monitoring.

<!-- ```bash
mamba create -n "isaaclab" "python==3.10"
mamba activate isaaclab
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com
``` -->

## Running the Benchmark

All scripts are provided in the scripts folder that you can simply run directly. Otherwise example usages are shown below for benchmarking simulation and simulation+rendering FPS. With --save-results flag on, resutls are saved to the `benchmark_results` folder in a .csv format. Running a benchmark with the same configurations of cameras/number of environments/choice of GPU will override the previous result. Example commands are shown below

### ManiSkill

```bash
python gpu_sim.py -e "CartpoleBalanceBenchmark-v1" \
    -n=2048 -o=state --save-results

python gpu_sim.py -e "CartpoleBalanceBenchmark-v1" \
    -n=1024 -o=rgb --num-cams=1 --cam-width=256 --cam-height=256 --save-results
```

### Isaac Lab

```bash
python isaac_lab_gpu_sim.py --task Isaac-Cartpole-Direct-Benchmark-v0 --headless \
    --num-envs=2048 --obs-mode=state --save-results

python isaac_lab_gpu_sim.py --task Isaac-Cartpole-RGB-Camera-Direct-Benchmark-v0 --headless \
    --num-cams=1 --cam-width=512 --cam-height=512 --enable_cameras \
    --num-envs=128 --obs-mode=rgb --save-results
```

<!-- ### Mujoco

```bash
python -m mujoco.mjx.testspeed --mjcf=envs/mujoco/panda_pick_cube.xml   --base_path=. --batch_size=4096 --nstep=100
``` -->