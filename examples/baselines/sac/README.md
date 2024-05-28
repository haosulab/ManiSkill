# Soft Actor Critic (SAC)



```bash
python sac.py --env_id="PushCube-v1" \
  --num_envs=16 --utd=0.5 \
  --total_timesteps=500_000 --eval_freq=50_000

python sac.py --env_id="PushCube-v1" \
  --num_envs=32 --utd=0.25 \
  --total_timesteps=500_000 --eval_freq=50_000
```