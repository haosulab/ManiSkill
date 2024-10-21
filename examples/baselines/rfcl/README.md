# Reverse Forward Curriculum Learning

Fast offline/online imitation learning from sparse rewards from very few demonstrations in simulation based on ["Reverse Forward Curriculum Learning for Extreme Sample and Demo Efficiency in Reinforcement Learning (ICLR 2024)"](https://arxiv.org/abs/2405.03379). Code adapted from https://github.com/StoneT2000/rfcl/

This code can be useful for solving tasks, verifying tasks are solvable via neural nets, generating infinite demonstrations via trained neural nets, all without using dense rewards and optionally without action labels.

## Installation
To get started run `git clone https://github.com/StoneT2000/rfcl.git rfcl_jax` which contains the code for RFCL written in jax. While ManiSkill3 does run on torch, the jax implementation is much more optimized and trains faster.

We recommend using conda/mamba and you can install the dependencies as so:

```bash
conda create -n "rfcl" "python==3.9"
conda activate rfcl
pip install "jax[cuda12_pip]==0.4.28" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e rfcl_jax
```

Then you can install ManiSkill and its dependencies

```bash
pip install mani_skill
pip install torch==2.3.1
```

We recommend installing the specific jax/torch versions in order to ensure they run correctly.

## Download and Process Dataset

Download demonstrations for a desired task e.g. PickCube-v1
```bash
python -m mani_skill.utils.download_demo "PickCube-v1"
```

<!-- TODO (stao): note how this part can be optional if user wants to do action free learning -->
Process the demonstrations in preparation for the learning workflow. We will use the teleoperated trajectories to train. Other provided demonstration sources (like motion planning and RL generated) can work as well but may require modifying a few hyperparameters. RFCL is extremely demonstration efficient and so we only need to process and save 5 demonstrations for training here.

```bash
env_id="PickCube-v1"
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/${env_id}/motionplanning/trajectory.h5 \
  --use-first-env-state \
  -c pd_joint_delta_pos -o state \
  --save-traj --count 5
```

## Train

To train with CPU vectorization (faster with a small number of parallel environments) with walltime efficient hyperparameters run

```bash
env_id=PickCube-v1
demos=5 # number of demos to train on
seed=42
XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py configs/base_sac_ms3.yml \
  logger.exp_name=rfcl-${env_id}-state-${demos}_motionplanning_demos-${seed}-walltime_efficient logger.wandb=True \
  seed=${seed} train.num_demos=${demos} train.steps=1_000_000 \
  env.env_id=${env_id} \
  train.dataset_path="~/.maniskill/demos/${env_id}/motionplanning/trajectory.state.pd_joint_delta_pos.cpu.h5"
```

You can add `train.train_on_demo_actions=False` to train on demonstrations without any action labels, just environment states. This may be useful if you can only download a dataset but can't convert the actions to the desired action space (some tasks can't easily convert actions)/

To train with sample efficient hyperparameters run

```bash
env_id=PickCube-v1
demos=5 # number of demos to train on
seed=42
XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py configs/base_sac_ms3_sample_efficient.yml \
  logger.exp_name=rfcl-${env_id}-state-${demos}_motionplanning_demos-${seed}-sample_efficient logger.wandb=True \
  seed=${seed} train.num_demos=${demos} train.steps=1_000_000 \
  env.env_id=${env_id} \
  train.dataset_path="~/.maniskill/demos/${env_id}/motionplanning/trajectory.state.pd_joint_delta_pos.cpu.h5"
```

Version of RFCL that runs on the GPU vectorized environments is currently not implemented as the current code is already quite fast and will require future research to investigate how to leverage GPU simulation with RFCL.


## Generating Demonstrations with Learned Policy 


To generate 1000 demonstrations with a trained policy you can run

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python rfcl_jax/scripts/collect_demos.py exps/path/to/model.jx \
  num_envs=8 num_episodes=1000
```

This saves the demos which uses CPU vectorization to generate demonstrations in parallel. The replay_trajectory tool can also be used to generate videos.

See the rfcl_jax/scripts/collect_demos.py code for details on how to load the saved policies and modify it to your needs.


## Citation

If you use this baseline please cite the following
```
@inproceedings{DBLP:conf/iclr/TaoSC024,
  author       = {Stone Tao and
                  Arth Shukla and
                  Tse{-}kai Chan and
                  Hao Su},
  title        = {Reverse Forward Curriculum Learning for Extreme Sample and Demo Efficiency},
  booktitle    = {The Twelfth International Conference on Learning Representations,
                  {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
  publisher    = {OpenReview.net},
  year         = {2024},
  url          = {https://openreview.net/forum?id=w4rODxXsmM},
  timestamp    = {Wed, 07 Aug 2024 17:11:53 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/TaoSC024.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```