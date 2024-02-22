# Soft Actor Critic

```bash
python sac.py --env-id="PushCube-v1" --num_envs=512 --total_timesteps=100000000
 python cleanrl_ppo_liftcube_state_gpu.py --num_envs=512 --gamma=0.8 --gae_lambda=0.9 --update_epochs=8 --target_kl=0.1 --num_minibatches=16  --env_id="PickCube-v0" --total_timesteps=100000000 --num_steps=100
```