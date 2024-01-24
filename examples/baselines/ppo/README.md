# Proximal Policy Optimization (PPO)

Code adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl/)

```bash
python ppo.py --num_envs=512 --gamma=0.8 --gae_lambda=0.9 --update_epochs=8 --target_kl=0.1 --num_minibatches=32 --env_id="PickCube-v1" --total_timesteps=100000000 --num_steps=100
python cleanrl_ppo_liftcube_state_gpu.py --num_envs=2048 --gamma=0.8 --gae_lambda=0.9 --update_epochs=1 --num_minibatches=32  --env_id="PushCube-v0" --total_timesteps=100000000 --num-steps=12
```