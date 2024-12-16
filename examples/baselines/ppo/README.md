# Proximal Policy Optimization (PPO)

Code for running the PPO RL algorithm is adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl/) and [LeanRL](https://github.com/pytorch-labs/LeanRL/). It is written to be single-file and easy to follow/read, and supports state-based RL and visual-based RL code.

Note that ManiSkill is still in beta, so we have not finalized training scripts for every pre-built task (some of which are simply too hard to solve with RL anyway).

Official baseline results can be run by using the scripts in the baselines.sh file. Results are organized and published to our [wandb report](https://api.wandb.ai/links/stonet2000/k6lz966q)

There is also now experimental support for PPO compiled and with CUDA Graphs enabled based on LeanRL. The code is in ppo_fast.py and you need to install [torchrl](https://github.com/pytorch/rl) and [tensordict](https://github.com/pytorch/tensordict/):

```bash
pip install torchrl tensordict
```

## State Based RL

Below is a sample of various commands you can run to train a state-based policy to solve various tasks with PPO that are lightly tuned already. The fastest one is the PushCube-v1 task which can take less than a minute to train on the GPU and the PickCube-v1 task which can take 2-5 minutes on the GPU.

The PPO baseline is not guaranteed to work for all tasks as some tasks do not have dense rewards yet or well tuned ones, or simply are too hard with standard PPO.


```bash
python ppo.py --env_id="PushCube-v1" \
  --num_envs=2048 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=2_000_000 --eval_freq=10 --num-steps=20
```

To evaluate, you can run
```bash
python ppo.py --env_id="PushCube-v1" \
   --evaluate --checkpoint=path/to/model.pt \
   --num_eval_envs=1 --num-eval-steps=1000
```

Note that with `--evaluate`, trajectories are saved from a GPU simulation. In order to support replaying these trajectories correctly with the `maniskill.trajectory.replay_trajectory` tool for some task, the number of evaluation environments must be fixed to `1`. This is necessary in order to ensure reproducibility for tasks that have randomizations on geometry (e.g. PickSingleYCB). Other tasks without geometrical randomization like PushCube are fine and you can increase the number of evaluation environments. 

The examples.sh file has a full list of tested commands for running state based PPO successfully on many tasks.

The results of running the baseline scripts for state based PPO are here: https://api.wandb.ai/links/stonet2000/k6lz966q.

## Visual (RGB) Based RL

Below is a sample of various commands for training a image-based policy with PPO that are lightly tuned. The fastest again is also PushCube-v1 which can take about 1-5 minutes and PickCube-v1 which takes 15-45 minutes. You will need to tune the `--num_envs` argument according to how much GPU memory you have as rendering visual observations uses a lot of memory. The settings below should all take less than 15GB of GPU memory. The examples.sh file has a full list of tested commands for running visual based PPO successfully on many tasks.


```bash
python ppo_rgb.py --env_id="PushCube-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=8 \
  --total_timesteps=1_000_000 --eval_freq=10 --num-steps=20
python ppo_rgb.py --env_id="PickCube-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=8 \
  --total_timesteps=10_000_000
python ppo_rgb.py --env_id="AnymalC-Reach-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=10_000_000 --num-steps=200 --num-eval-steps=200 \
  --gamma=0.99 --gae_lambda=0.95
```

To evaluate a trained policy you can run

```bash
python ppo_rgb.py --env_id="PickCube-v1" \
  --evaluate --checkpoint=path/to/model.pt \
  --num_eval_envs=1 --num-eval-steps=1000
```

and it will save videos to the `path/to/test_videos`.

The examples.sh file has a full list of tested commands for running RGB based PPO successfully on many tasks.

The results of running the baseline scripts for RGB based PPO are here: https://api.wandb.ai/links/stonet2000/k6lz966q

## Visual (RGB+Depth) Based RL

WIP

## Visual (Pointcloud) Based RL

WIP

## Replaying Evaluation Trajectories

It might be useful to get some nicer looking videos. A simple way to do that is to first use the evaluation scripts provided above. It will then save a .h5 and .json file with a name equal to the date and time that you can then replay with different settings as so

```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path=path/to/trajectory.h5 --use-env-states --shader="rt-fast" \
  --save-video --allow-failure -o "none"
```

This will use environment states to replay trajectories, turn on the ray-tracer (There is also "rt" which is higher quality but slower), and save all videos including failed trajectories.

## Some Notes

- Evaluation with GPU simulation (especially with randomized objects) is a bit tricky. We recommend reading through [our docs](https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/baselines.html#evaluation) on online RL evaluation in order to understand how to fairly evaluate policies with GPU simulation.
- Many tasks support visual observations, however we have not carefully verified yet if the camera poses for the tasks are setup in a way that makes it possible to solve some tasks from visual observations.

## Citation

If you use this baseline please cite the following
```
@article{DBLP:journals/corr/SchulmanWDRK17,
  author       = {John Schulman and
                  Filip Wolski and
                  Prafulla Dhariwal and
                  Alec Radford and
                  Oleg Klimov},
  title        = {Proximal Policy Optimization Algorithms},
  journal      = {CoRR},
  volume       = {abs/1707.06347},
  year         = {2017},
  url          = {http://arxiv.org/abs/1707.06347},
  eprinttype    = {arXiv},
  eprint       = {1707.06347},
  timestamp    = {Mon, 13 Aug 2018 16:47:34 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/SchulmanWDRK17.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```