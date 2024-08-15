# Diffusion Policy

Code for running the Diffusion Policy algorithm based on ["Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"](https://arxiv.org/abs/2303.04137v4). It is adapted from the [original code](https://github.com/real-stanford/diffusion_policy).

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

## State-Based Training

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

## Citation

If you use this baseline please cite the following
```
@inproceedings{DBLP:conf/rss/ChiFDXCBS23,
  author       = {Cheng Chi and
                  Siyuan Feng and
                  Yilun Du and
                  Zhenjia Xu and
                  Eric Cousineau and
                  Benjamin Burchfiel and
                  Shuran Song},
  editor       = {Kostas E. Bekris and
                  Kris Hauser and
                  Sylvia L. Herbert and
                  Jingjin Yu},
  title        = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  booktitle    = {Robotics: Science and Systems XIX, Daegu, Republic of Korea, July
                  10-14, 2023},
  year         = {2023},
  url          = {https://doi.org/10.15607/RSS.2023.XIX.026},
  doi          = {10.15607/RSS.2023.XIX.026},
  timestamp    = {Mon, 29 Apr 2024 21:28:50 +0200},
  biburl       = {https://dblp.org/rec/conf/rss/ChiFDXCBS23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```