# Soft Actor Critic (SAC)

Code for running the PPO RL algorithm is adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl/) and our previous [ManiSkill Baselines](https://github.com/tongzhoumu/ManiSkill_Baselines/). It is written to be single-file and easy to follow/read, and supports state-based RL and visual-based RL code.

Note that ManiSkill is still in beta, so we have not finalized training scripts for every pre-built task (some of which are simply too hard to solve with RL anyway). We will further organize these scripts and results in a more organized manner in the future.

## State Based RL


```bash
python sac.py --env_id="PushCube-v1" \
  --num_envs=32 --utd=0.5 \
  --total_timesteps=200_000 --eval_freq=50_000
```