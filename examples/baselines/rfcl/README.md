# Reverse Forward Curriculum Learning

Fast offline/online imitation learning from sparse rewards in simulation based on ["Reverse Forward Curriculum Learning for Extreme Sample and Demo Efficiency in Reinforcement Learning (ICLR 2024)"](https://arxiv.org/abs/2405.03379). Code adapted from https://github.com/StoneT2000/rfcl/

This code can be useful for solving tasks, verifying tasks are solvable via neural nets, and generating infinite demonstrations via trained neural nets, all without using dense rewards (provided the task is not too long horizon)

## Installation
To get started run `git clone https://github.com/StoneT2000/rfcl.git rfcl_jax --branch ms3-gpu` which contains the code for RFCL written in jax. While ManiSkill3 does run on torch, the jax implementation is much more optimized and trains faster.

We recommend using conda/mamba and you can install the dependencies as so:

```bash
conda create -n "rfcl" "python==3.9"
conda activate rfcl
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e rfcl_jax
```

Then you can install ManiSkill and its dependencies

```bash
pip install mani_skill
pip install torch
```

## Download and Process Dataset

Download demonstrations for a desired task e.g. PickCube-v1
```bash
python -m mani_skill.utils.download_demo "PickCube-v1"
```

<!-- TODO (stao): note how this part can be optional if user wants to do action free learning -->
Process the demonstrations in preparation for the learning workflow. We will use the teleoperated trajectories to train. Other provided demonstration sources (like motion planning and RL generated) can work as well but may require modifying a few hyperparameters.
```bash
env_id="PickCube-v1"
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/${env_id}/motionplanning/trajectory.h5 \
  --use-first-env-state \
  -c pd_joint_delta_pos -o state \
  --save-traj
```

## Train

To train with CPU vectorization (faster with a small number of parallel environments) run

```bash
env_id=PickCube-v1
demos=5 # number of demos to train on
seed=42
XLA_PYTHON_CLIENT_PREALLOCATE=false python rfcl_jax/train.py configs/base_sac_ms3.yml \
  logger.exp_name="ms3/${env_id}/rfcl_${demos}_demos_s${seed}" logger.wandb=True \
  seed=${seed} train.num_demos=${demos} train.steps=1_000_000 \
  env.env_id=${env_id} \
  train.dataset_path="~/.maniskill/demos/${env_id}/motionplanning/trajectory.state.pd_joint_delta_pos.h5" 
```

Version of RFCL that runs on the GPU vectorized environments is currently not implemented as the current code is already quite fast

<!-- To train with the GPU vectorization TODO (stao):
```bash
env=pickcube
demos=5
seed=1
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_ms3.py rfcl_jax/configs/ms3-gpu/sac_ms3_${env}.yml \
  logger.exp_name="ms3/${env}/${name_prefix}_${demos}_demos_s${seed}-gpusim" \
  logger.wandb=False \
  train.num_demos=${demos} \
  seed=${seed} \
  train.steps=4000000
``` -->


## Generating Demonstrations with Learned Policy 


To generate 1000 demonstrations you can run

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python rfcl_jax/scripts/collect_demos.py exps/path/to/model.jx \
  num_envs=8 num_episodes=1000
```
This saves the demos 

which uses CPU vectorization to generate demonstrations in parallel. Note that while the demos are generated on the CPU, you can always convert them to demonstrations on the GPU via the [replay trajectory tool](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html) as so

```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path exps/ms3/PickCube-v1/_5_demos_s42/eval_videos/trajectory.h5 \
  -b gpu --use-first-env-state
```