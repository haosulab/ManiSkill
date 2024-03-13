# Proximal Policy Optimization (PPO)

Code adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl/)

Below is a sample of various commands you can run to train a policy to solve various tasks with PPO that are lightly tuned already. The fastest one is the PushCube-v1 task which can take less than a minute to train on the GPU.

```bash
python ppo.py --env_id="PushCube-v1" \
  --num_envs=2048 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=5_000_000 --eval_freq=10 --num-steps=20
python ppo.py --env_id="PickCube-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=10_000_000
python ppo.py --env_id="StackCube-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=25_000_000
python ppo.py --env_id="PickSingleYCB-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=25_000_000
python ppo.py --env_id="TwoRobotStackCube-v1" \
   --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=40_000_000 --num-steps=100 --num-eval-steps=100
```

<!-- TODO (stao, arnav) clean up the baseline code to be slightly nicer (track FPS, and update time separately), and put results onto wandb. Also merge code with CleanRL Repo (stao can ask costa to help do that) -->