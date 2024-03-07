# Reinforcement Learning with ManiSkill

This contains single-file implementations that solve with LiftCube environment with rgbd or state observations. You need to install Stable Baselines 3 as so to run it. 

```
pip install --upgrade stable_baselines3 
```

All scripts contain the same arguments and can be run as so

```
# Training
python sb3_ppo_liftcube_rgbd.py

# Evaluation
python sb3_ppo_liftcube_rgbd.py --eval --model-path=path/to/model
````

Pass in `--help` for more options (e.g. logging, number of parallel environments, whether to use ManiSkill Vectorized Environments or not etc.). Models and videos are saved to the folder specified by `--log-dir` which defaults to `logs/`. 



```bash
python cleanrl_ppo_liftcube_state_gpu.py --num_envs=512 --gamma=0.8 --gae_lambda=0.9 --update_epochs=8 --target_kl=0.1 --num_minibatches=16 --env_id="PickCube-v0" --total_timesteps=100000000 --num_steps=100
```