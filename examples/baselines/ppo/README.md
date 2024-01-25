# Proximal Policy Optimization (PPO)

Code adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl/)

```bash
python ppo.py --num_envs=512 --update_epochs=8 --target_kl=0.1 --num_minibatches=32 --env_id="PickCube-v1" --total_timesteps=100000000
python ppo.py --num_envs=2048 --update_epochs=1 --num_minibatches=32  --env_id="PushCube-v1" --total_timesteps=100000000 --num-steps=12
```