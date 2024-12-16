# Reinforcement Learning from Prior Data (RLPD)

Sample-efficient offline/online imitation learning from sparse rewards leveraging prior data based on ["Efficient Online Reinforcement Learning with Offline Data
(ICML 2023)"](https://arxiv.org/abs/2302.02948). Code adapted from https://github.com/ikostrikov/rlpd

RLPD leverages prior collected trajectory data (expert and non-expert work) and trains on the prior data while collecting online data to sample efficiently learn a policy to solve a task with just sparse rewards.

## Installation

To get started run `git clone https://github.com/StoneT2000/rfcl.git rlpd_jax` which contains the code for RLPD written in jax (a partial fork of the original RLPD and JaxRL repos that has been optimized to run faster and support vectorized environments).

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

To train with environment vectorization run

```bash
env_id=PickCube-v1
demos=1000 # number of demos to train on.
seed=42
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_ms3.py configs/base_rlpd_ms3.yml \
  logger.exp_name="rlpd-${env_id}-state-${demos}_rl_demos-${seed}-walltime_efficient" logger.wandb=True \
  seed=${seed} train.num_demos=${demos} train.steps=200_000 \
  env.env_id=${env_id} \
  train.dataset_path="~/.maniskill/demos/${env_id}/rl/trajectory.state.pd_joint_delta_pos.cpu.h5"
```

This should solve the PickCube-v1 task in a few minutes, but won't get good sample efficiency.

For sample-efficient settings you can use the sample-efficient configurations stored in configs/base_rlpd_ms3_sample_efficient.yml (no env parallelization, more critics, higher update-to-data ratio). This will take less environment samples (around 50K to solve) but runs slower.

```bash
env_id=PickCube-v1
demos=1000 # number of demos to train on.
seed=42
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_ms3.py configs/base_rlpd_ms3_sample_efficient.yml \
  logger.exp_name="rlpd-${env_id}-state-${demos}_rl_demos-${seed}-sample_efficient" logger.wandb=True \
  seed=${seed} train.num_demos=${demos} train.steps=100_000 \
  env.env_id=${env_id} \
  train.dataset_path="~/.maniskill/demos/${env_id}/rl/trajectory.state.pd_joint_delta_pos.cpu.h5"
```

evaluation videos are saved to `exps/<exp_name>/videos`.

## Generating Demonstrations / Evaluating policies

To generate 1000 demonstrations with a trained policy you can run

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python rlpd_jax/scripts/collect_demos.py exps/path/to/model.jx \
  num_envs=8 num_episodes=1000
```
This saves the demos which uses CPU vectorization to generate demonstrations in parallel. The replay_trajectory tool can also be used to generate videos.

See the rlpd_jax/scripts/collect_demos.py code for details on how to load the saved policies and modify it to your needs.

## Citation

If you use this baseline please cite the following
```
@inproceedings{DBLP:conf/icml/BallSKL23,
  author       = {Philip J. Ball and
                  Laura M. Smith and
                  Ilya Kostrikov and
                  Sergey Levine},
  editor       = {Andreas Krause and
                  Emma Brunskill and
                  Kyunghyun Cho and
                  Barbara Engelhardt and
                  Sivan Sabato and
                  Jonathan Scarlett},
  title        = {Efficient Online Reinforcement Learning with Offline Data},
  booktitle    = {International Conference on Machine Learning, {ICML} 2023, 23-29 July
                  2023, Honolulu, Hawaii, {USA}},
  series       = {Proceedings of Machine Learning Research},
  volume       = {202},
  pages        = {1577--1594},
  publisher    = {{PMLR}},
  year         = {2023},
  url          = {https://proceedings.mlr.press/v202/ball23a.html},
  timestamp    = {Mon, 28 Aug 2023 17:23:08 +0200},
  biburl       = {https://dblp.org/rec/conf/icml/BallSKL23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```