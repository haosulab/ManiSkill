# Diffusion Policy

Code for running the Diffusion Policy algorithm is adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl/). It is written to be single-file and easy to follow/read, and supports state-based RL and visual-based RL code.

## Installation

To get started, we recommend using conda/mamba to create a new environment and install the dependencies

```bash
conda create -n diffusion-policy-ms python=3.9
conda activate diffusion-policy-ms
pip install diffusers tensorboard wandb
```

Then you can install ManiSkill

```bash
pip install mani_skill
```

## Demonstration Download and Preprocessing

```
env_id="PickCube-v1"
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/${env_id}/motionplanning/trajectory.h5 \
  --use-first-env-state \
  -c pd_joint_delta_pos -o state \
  --save-traj
```


```bash
python train.py -e "PickCube-v1" --dataset_path "~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5"
```
## Evaluation
