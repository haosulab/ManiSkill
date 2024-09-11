# Performance Benchmarking

See [the performance benchmarking documentation](https://maniskill.readthedocs.io/en/latest/user_guide/additional_resources/performance_benchmarking.html) for in depth details.

If you plan to run the code here you need to git clone ManiSkill and change your directory to this one before running the code

Code Structure:
- `benchmark.sh`: Bash scripts to run a matrix of performance tests. Results are saved to a local `benchmark_results` folder
- `plot_results.py`: Run this code to generate graphs of performance results saved to `benchmark_results`
- `envs/`: custom environments built for benchmarking, designed to be as close as possible between different simulators


## Setup

### ManiSkill

See https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html

### Mujoco / MJX

```bash
mamba create -n "mujoco_benchmarking" "python==3.11"
mamba activate mujoco_benchmarking
pip install mujoco-mjx
pip install "jax[cuda12]"
```

## Running the Benchmark


### Mujoco

```bash
python -m mujoco.mjx.testspeed --mjcf=envs/mujoco/panda_pick_cube.xml   --base_path=. --batch_size=4096 --nstep=100
```