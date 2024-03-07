# Decision Transformer

This baseline is adapted from the `gym` example provided [here](https://github.com/kzl/decision-transformer/tree/master). This folder provides both a jupyter notebook version that can be run on Google Colab as well as a python script that can be used for training a model. A few environments have already been trained on and these runs can be viewed [here](https://wandb.ai/ag115115/decision-transformer).

## Run Training Using Python Script

### Using Docker

1. Pull the docker image

   `docker pull ag2897387/dec_tr_mani:latest`

2. Run the training script

   `docker run --rm -it --gpus all ag2897387/dec_tr_mani:latest python train.py --env LiftCube-v0 --dataset demos/v0/rigid_body/LiftCube-v0/trajectory.state.pd_ee_delta_pose.h5`

There are additional arguments that specify parameters for the model and such.

Results can also be logged to [Wandb](https://wandb.ai/) using the follow arguments.

`docker run --rm -it --gpus all ag2897387/dec_tr_mani python train.py --dataset demos/v0/rigid_body/{env_id}/trajectory.state.pd_ee_delta_pose.h5 --log_to_wandb true --wandb_key <insert_wandb_key>`

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
