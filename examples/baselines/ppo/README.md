# Proximal Policy Optimization (PPO)

Code for running the PPO RL algorithm is adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl/). It is written to be a single-file and easy to follow/read

To train, you can run

```bash
python ppo.py --env_id="PushCube-v1" \
  --num_envs=2048 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=5_000_000 --eval_freq=10 --num-steps=20
```

To evaluate, you can run
```bash
python ppo.py --env_id="PickCube-v1" \
   --evaluate --num_eval_envs=1 --checkpoint=runs/PickCube-v1__ppo__1__1710225023/ppo_101.cleanrl_model
```

Note that with `--evaluate`, trajectories are saved from a GPU simulation. In order to support replaying these trajectories correctly with the `maniskill.trajectory.replay_trajectory` tool, the number of evaluation environments must be fixed to `1`. This is necessary in order to ensure reproducibility for tasks that have randomizations on geometry (e.g. PickSingleYCB).


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
  --total_timesteps=150_000_000 --num-steps=100 --num-eval-steps=100
python ppo.py --env_id="TwoRobotStackCube-v1" \
   --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=40_000_000 --num-steps=100 --num-eval-steps=100
python ppo.py --env_id="RotateCubeLevel0-v1" \
   --num_envs=128 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=50_000_000 --num-steps=250 --num-eval-steps=250
python ppo.py --env_id="RotateCubeLevel1-v1" \
   --num_envs=128 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=50_000_000 --num-steps=250 --num-eval-steps=250
python ppo.py --env_id="RotateCubeLevel2-v1" \
   --num_envs=128 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=50_000_000 --num-steps=250 --num-eval-steps=250
python ppo.py --env_id="RotateCubeLevel3-v1" \
   --num_envs=128 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=50_000_000 --num-steps=250 --num-eval-steps=250
python ppo.py --env_id="RotateCubeLevel4-v1" \
   --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=500_000_000 --num-steps=250 --num-eval-steps=250
```