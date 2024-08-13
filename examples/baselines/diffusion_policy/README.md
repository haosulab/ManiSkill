# Diffusion Policy

Code for running the Diffusion Policy algorithm is adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl/). It is written to be single-file and easy to follow/read, and supports state-based RL and visual-based RL code.

## Installation

To get started, we recommend using conda/mamba to create a new environment and install the dependencies

```bash
conda create -n diffusion-policy-ms python=3.9
conda activate diffusion-policy-ms
pip install -e .
```

## Demonstration Download and Preprocessing

By default for fast downloads and smaller file sizes, ManiSkill demonstrations are stored in a highly reduced/compressed format which includes not keeping any observation data. Run the command to download the demonstration and convert it to a format that includes observation data and the desired action space.

```bash
python -m mani_skill.utils.download_demo "PickCube-v1"
```

```bash
env_id="PickCube-v1"
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/${env_id}/motionplanning/trajectory.h5 \
  --use-first-env-state \
  -c pd_joint_delta_pos -o state \
  --save-traj --num-procs 10
```

## Training

Example training, learning from 100 demonstrations generated via motionplanning in the PickCube-v1 task
```bash
seed=42
demos=100
env_id="PickCube-v1"
python train.py --env-id ${env_id} --max_episode_steps 100 --total_iters 30000 \
  --control-mode "pd_joint_delta_pos" --num-demos ${demos} --seed ${seed} \
  --demo-path ~/.maniskill/demos/${env_id}/motionplanning/trajectory.state.pd_joint_delta_pos.cpu.h5 \
  --exp-name diffusion_policy-${env_id}-state-${demos}_motionplanning_demos-${seed} \
  --demo_type="motionplanning" --track # additional tag for logging purposes on wandb
```
In tensorboard/wandb there are two success rates reported, `success_once` and `success_at_end`. `success_once` considers success when the episode achieves success at any point in the episode, and `success_at_end` considers success only when the episode achieves success at the last step after `max_episode_steps` are reached.

Note that we further add a `--max_episode_steps` argument to the training script to allow for longer demonstrations to be learned from (such as motionplanning / teleoperated demonstrations). By default the max episode steps of most environments are tuned lower so reinforcement learning agents can learn faster. You may need to increase this value depending on the task and the demonstrations you are using. 

## Train and Evaluate with GPU Simulation

You can also choose to train on trajectories generated in the GPU simulation and evaluate faster with the GPU simulation. However as most demonstrations are usually generated in the CPU simulation (via motionplanning or teleoperation), you may observe worse performance when evaluating on the GPU simulation vs the CPU simulation.

It is also recommended to not save videos if you are using a lot of parallel environments as the video size can get very large.

```bash
seed=42
demos=100
python train.py --env-id ${env_id} --max_episode_steps 100 --total_iters 30000 \
  --control-mode "pd_joint_delta_pos" --num-demos ${demos} --seed ${seed} \
  --demo-path ~/.maniskill/demos/${env_id}/motionplanning/trajectory.state.pd_joint_delta_pos.cuda.h5 \
  --exp-name diffusion_policy-${env_id}-state-${demos}_motionplanning_demos-${seed} \
  --sim-backend="gpu" --num-eval-envs 100 --no-capture-video \
  --demo_type="motionplanning" --track # additional tag for logging purposes on wandb
```