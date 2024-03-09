# Diffusion Policy

This baseline is adapted from the example provided [here](https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B). This folder provides both a jupyter notebook version that can be run on Google Colab as well as a python script that can be used for training a model. A few environments have already been trained on and these runs can be viewed [here](https://wandb.ai/ag115115/diffusion-policy).

## Run Training Using Python Script

### Using Docker

1. Pull the docker image

   `docker pull ag2897387/diff_policy_maniskill:latest`

2. Run the training script

   `docker run --rm -it --gpus all ag2897387/diff_policy_maniskill python train.py --env LiftCube-v0 --dataset demos/v0/rigid_body/LiftCube-v0/trajectory.state.pd_ee_delta_pose.h5`

There are additional arguments that specify parameters for the model and such.

Results can also be logged to [Wandb](https://wandb.ai/) using the follow arguments.

`docker run --rm -it --gpus all ag2897387/diff_policy_maniskill python train.py --env LiftCube-v0 --dataset demos/v0/rigid_body/LiftCube-v0/trajectory.state.pd_ee_delta_pose.h5 --log_to_wandb true --wandb_key <insert_wandb_key>`

Additional arguments can be seen in train.py.

### Without Docker

1. Install dependencies

```
pip install mani_skill
pip install -r requirements.txt
```

2. Run the training script

```
python train.py --env LiftCube-v0 --dataset demos/v0/rigid_body/LiftCube-v0/trajectory.state.pd_ee_delta_pose.h5
```
