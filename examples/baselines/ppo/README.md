# Proximal Policy Optimization (PPO)

Code for running the PPO RL algorithm is adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl/). It is written to be single-file and easy to follow/read, and supports state-based RL and visual-based RL code.

Note that ManiSkill is still in beta, so we have not finalized training scripts for every pre-built task (some of which are simply too hard to solve with RL anyway). We will further organize these scripts and results in a more organized manner in the future.


## State Based RL

Below is a sample of various commands you can run to train a state-based policy to solve various tasks with PPO that are lightly tuned already. The fastest one is the PushCube-v1 task which can take less than a minute to train on the GPU and the PickCube-v1 task which can take 2-5 minutes on the GPU.

The PPO baseline is not guaranteed to work for tasks not tested below as some tasks do not have dense rewards yet or well tuned ones, or simply are too hard with standard PPO (or our team has not had time to verify results yet)


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


Below is a full list of various commands you can run to train a policy to solve various tasks with PPO that are lightly tuned already. The fastest one is the PushCube-v1 task which can take less than a minute to train on the GPU.

```bash

python ppo.py --env_id="PickCube-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=10_000_000
python ppo.py --env_id="StackCube-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=25_000_000
python ppo.py --env_id="PickSingleYCB-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=25_000_000
python ppo.py --env_id="PegInsertionSide-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=250_000_000 --num-steps=100 --num-eval-steps=100
python ppo.py --env_id="TwoRobotStackCube-v1" \
   --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=40_000_000 --num-steps=100 --num-eval-steps=100
python ppo.py --env_id="TriFingerRotateCubeLevel0-v1" \
   --num_envs=128 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=50_000_000 --num-steps=250 --num-eval-steps=250
python ppo.py --env_id="TriFingerRotateCubeLevel1-v1" \
   --num_envs=128 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=50_000_000 --num-steps=250 --num-eval-steps=250
python ppo.py --env_id="TriFingerRotateCubeLevel2-v1" \
   --num_envs=128 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=50_000_000 --num-steps=250 --num-eval-steps=250
python ppo.py --env_id="TriFingerRotateCubeLevel3-v1" \
   --num_envs=128 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=50_000_000 --num-steps=250 --num-eval-steps=250
python ppo.py --env_id="TriFingerRotateCubeLevel4-v1" \
   --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=500_000_000 --num-steps=250 --num-eval-steps=250

python ppo.py --env_id="MS-CartpoleBalance-v1" \
   --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=4_000_000 --num-steps=250 --num-eval-steps=1000 \
   --gamma=0.99 --gae_lambda=0.95 \
   --eval_freq=5

python ppo.py --env_id="MS-CartpoleSwingUp-v1" \
   --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=10_000_000 --num-steps=250 --num-eval-steps=1000 \
   --gamma=0.99 --gae_lambda=0.95 \
   --eval_freq=5

# note that quadruped simulation is still being optimized and the reward function for this task
# is not that optimal although it works.
python ppo.py --env_id="AnymalC-Reach-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=75_000_000 --num-steps=200 --num-eval-steps=200 \
  --gamma=0.99 --gae_lambda=0.95
python ppo.py --env_id="UnitreeH1Stand-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=100_000_000 --num-steps=100 --num-eval-steps=1000 \
  --gamma=0.99 --gae_lambda=0.95

python ppo.py --env_id="OpenCabinetDrawer-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=10_000_000 --num-steps=100 --num-eval-steps=100   
```

## Visual (RGB) Based RL

Below is a sample of various commands for training a image-based policy with PPO that are lightly tuned. The fastest again is also PushCube-v1 which can take about 1-5 minutes and PickCube-v1 which takes 15-45 minutes. You will need to tune the `--num_envs` argument according to how much GPU memory you have as rendering visual observations uses a lot of memory. The settings below should all take less than 15GB of GPU memory. Note that while if you have enough memory you can easily increase the number of environments, this does not necessarily mean wall-time or sample efficiency improve.

The visual PPO baseline is not guaranteed to work for tasks not tested below as some tasks do not have dense rewards yet or well tuned ones, or simply are too hard with standard PPO (or our team has not had time to verify results yet).



```bash
python ppo_rgb.py --env_id="PushCube-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=8 \
  --total_timesteps=1_000_000 --eval_freq=10 --num-steps=20
python ppo_rgb.py --env_id="PickCube-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=8 \
  --total_timesteps=10_000_000
python ppo_rgb.py --env_id="PickSingleYCB-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=8 \
  --total_timesteps=10_000_000
```

To evaluate a trained policy you can run

```bash
python ppo_rgb.py --env_id="PickCube-v1" \
  --evaluate --checkpoint=path/to/model.pt \
  --num_eval_envs=1 --num-eval-steps=1000
```

and it will save videos to the `path/to/test_videos`.

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

- The code currently does not have the best way to evaluate the agents in that during GPU simulation, all assets are frozen per parallel environment (changing them slows training down). Thus when doing evaluation, even though we evaluate on multiple (8 is default) environments at once, they will always feature the same set of geometry. This only affects tasks where there is geometry variation (e.g. PickClutterYCB, OpenCabinetDrawer). You can make it more accurate by increasing the number of evaluation environments. Our team is discussing still what is the best way to evaluate trained agents properly without hindering performance.
- Many tasks support visual observations, however we have not carefully verified if the camera poses for the tasks are setup in a way that makes it possible to solve the task from visual observations.

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