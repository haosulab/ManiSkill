# Reinforcement Learning from Prior Data (RLPD)

Sample-efficient offline/online imitation learning from sparse rewards leveraging prior data based on ["Efficient Online Reinforcement Learning with Offline Data
(ICML 2023)"](https://arxiv.org/abs/2302.02948). Code adapted from https://github.com/ikostrikov/rlpd

RLPD leverages prior collected trajectory data (expert and non-expert work) and trains on the prior data while collecting online data to sample efficiently learn a policy to solve a task with just sparse rewards.

## Installation

To get started run `git clone https://github.com/StoneT2000/rfcl.git rlpd_jax --branch ms3-gpu` which contains the code for RLPD written in jax (a partial fork of the original RLPD and JaxRL repos that has been optimized to run faster and support vectorized environments).

We recommend using conda/mamba and you can install the dependencies as so:

```bash
conda create -n "rlpd_ms3" "python==3.9"
conda activate rlpd_ms3
pip install --upgrade "jax[cuda12_pip]==0.4.28" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e rlpd_jax
```

Then you can install ManiSkill and its dependencies

```bash
pip install mani_skill torch==2.3.1
```
Note that since jax and torch are used, we recommend installing the specific versions detailed in the commands above as those are tested to work together.

## Download and Process Dataset

Download demonstrations for a desired task e.g. PickCube-v1
```bash
python -m mani_skill.utils.download_demo "PickCube-v1"
```

<!-- TODO (stao): note how this part can be optional if user wants to do action free learning -->
Process the demonstrations in preparation for the learning workflow. RLPD works well for harder tasks if sufficient data is provided and the data itself is not too multi-modal. Hence we will use the RL generated trajectories (lot of data and not multi-modal so much easier to learn) for the example below:


The preprocessing step here simply replays all trajectories by environment state (so the exact same trajectory is returned) and save the state observations to train on. Moreover failed demos are also saved as RLPD can learn from sub-optimal data as well.

```bash
env_id="PickCube-v1"
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/${env_id}/rl/trajectory.h5 \
  --use-env-state --allow-failure \
  -c pd_joint_delta_pos -o state \
  --save-traj --num-procs 4
```

## Train

To train with environment vectorization (wall-time fast settings) run

```bash
env_id=PickCube-v1
demos=5 # number of demos to train on.
seed=42
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_ms3.py configs/base_rlpd_ms3.yml \
  logger.exp_name="ms3/${env_id}/rlpd_${demos}_demos_s${seed}" logger.wandb=True \
  seed=${seed} train.num_demos=${demos} train.steps=5_000_000 \
  env.env_id=${env_id} \
  train.dataset_path="~/.maniskill/demos/${env_id}/rl/trajectory.state.pd_joint_delta_pos.h5" 
```

This should solve the PickCube-v1 task in a few minutes.