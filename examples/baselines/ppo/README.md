# Proximal Policy Optimization (PPO)

Code adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl/)

Below is a sample of various commands you can run to train a policy to solve various tasks with PPO that are lightly tuned already. The fastest one is the PushCube-v1 task which can take less than a minute to train on the GPU.

```bash
python ppo.py --num_envs=2048 --update_epochs=8 --num_minibatches=32  --env_id="PushCube-v1" --total_timesteps=10000000 --eval_freq=10
python ppo.py --num_envs=1024 --update_epochs=8 --num_minibatches=32 --env_id="PickCube-v1" --total_timesteps=50000000
python ppo.py --num_envs=1024 --update_epochs=8 --num_minibatches=32 --env_id="StackCube-v1" --total_timesteps=100000000
python ppo.py --num_envs=512 --update_epochs=8 --num_minibatches=32 --env_id="TwoRobotStackCube-v1" --total_timesteps=100000000 --num-steps=100
```