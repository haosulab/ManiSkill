# Soft Actor Critic (SAC)



```bash
python sac.py --env_id="PushCube-v1" \
  --num_envs=16 --utd=0.5 --num_minibatches= \
  --total_timesteps=100_000 --eval_freq=50_000
```