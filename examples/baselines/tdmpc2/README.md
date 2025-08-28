# Temporal Difference Learning for Model Predictive Control 2 (TD-MPC2)

Scalable, robust model-based RL algorithm based on ["TD-MPC2: Scalable, Robust World Models for Continuous Control"](https://arxiv.org/abs/2310.16828). Code adapted from https://github.com/nicklashansen/tdmpc2. It is written to work with the new Maniskill update, and supports vectorized state-based and visual-based RL environment.

## Installation
We recommend using conda/mamba and you can install the dependencies as so :

```bash
conda env create -f environment.yaml
conda activate tdmpc2-ms
```

or follow the [original repo](https://github.com/nicklashansen/tdmpc2)'s guide to build the docker image.


## State Based RL

Simple command to run the algorithm with default configs (5M params, 1M steps, default control mode, 32 envs, state obs mode) :
```bash
python train.py env_id=PushCube-v1
```

More advanced command with optional configs : (More can be found in config.yaml)
```bash
python train.py model_size=5 steps=1_000_000 seed=1 exp_name=default \
  env_id=PushCube-v1 env_type=gpu num_envs=32 control_mode=pd_ee_delta_pose obs=state \
  save_video_local=false wandb=true wandb_entity=??? wandb_project=??? wandb_group=??? wandb_name=??? setting_tag=??? 
```
(*) The optional *setting_tag* is for adding a specific tag in the wandb log (e.g. sample_efficient, walltime_efficient, etc.)

## Visual (RGB) Based RL

The visual based RL expects model_size = 5. Also, make sure you have sufficient CPU memory, otherwise lower the buffer_size and use gpu env.
```bash
python train.py buffer_size=500_000 steps=5_000_000 seed=1 exp_name=default \
  env_id=PushCube-v1 env_type=gpu num_envs=32 control_mode=pd_ee_delta_pose obs=rgb \
  save_video_local=false wandb=true wandb_entity=??? wandb_project=??? wandb_group=??? wandb_name=??? setting_tag=???
```

## Replaying Evaluation Trajectories

To create videos of a checkpoint model, use the following command.

```bash
python evaluate.py model_size=5 seed=1 exp_name=default \ 
  env_id=PushCube-v1 control_mode=pd_ee_delta_pose obs=state \
  save_video_local=true checkpoint=/absolute/path/to/checkpoint.pt
```

* **Make sure you specify the same `num_eval_envs` and `control_mode` the model was trained on if it's not default.**
* The video are saved under ```logs/{env_id}/{seed}/{exp_name}/videos```
* The number of video saved is determined by ```num_eval_envs * eval_episodes_per_env```

## Some Notes

- Multi-task TD-MPC2 isn't supported for Maniskill at the moment.

## Citation

If you use this baseline please cite the following
```
@inproceedings{hansen2024tdmpc2,
  title={TD-MPC2: Scalable, Robust World Models for Continuous Control}, 
  author={Nicklas Hansen and Hao Su and Xiaolong Wang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```
as well as the original TD-MPC paper:
```
@inproceedings{hansen2022tdmpc,
  title={Temporal Difference Learning for Model Predictive Control},
  author={Nicklas Hansen and Xiaolong Wang and Hao Su},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2022}
}
```