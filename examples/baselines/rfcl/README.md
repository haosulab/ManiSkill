# Reverse Forward Curriculum Learning

Fast offline/online imitation learning in simulation based on "Reverse Forward Curriculum Learning for Extreme Sample and Demo Efficiency in Reinforcement Learning (ICLR 2024)". Code adapted from https://github.com/StoneT2000/rfcl/

## Installation
To get started run `git clone https://github.com/StoneT2000/rfcl.git rfcl_jax --branch ms3-gpu` which contains the code for RFCL written in jax.

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

Process the demonstrations in preparation for the imitation learning workflow
```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/teleop/trajectory.h5 \
  --use-first-env-state \
  -c pd_joint_delta_pos -o state \
  --save-traj
```

## Train

To train with CPU vectorization (faster with a small number of parallel environments) run

```bash
env=pickcube
demos=5
seed=0
XLA_PYTHON_CLIENT_PREALLOCATE=false python rfcl_jax/train.py rfcl_jax/configs/ms3/sac_ms3_${env}.yml \
  logger.exp_name="ms3/${env}/${name_prefix}_${demos}_demos_s${seed}" \
  logger.wandb=False \
  train.num_demos=${demos} \
  seed=${seed} \
  train.steps=4000000
```

To train with the GPU vectorization (faster with a large number of parallel environments) run
```bash
env=pickcube
demos=5
seed=0
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_ms3.py rfcl_jax/configs/ms3/sac_ms3_${env}.yml \
  logger.exp_name="ms3/${env}/${name_prefix}_${demos}_demos_s${seed}" \
  logger.wandb=False \
  train.num_demos=${demos} \
  seed=${seed} \
  train.steps=4000000
```