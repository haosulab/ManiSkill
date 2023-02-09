# Reinforcement Learning with ManiSkill2

This contains single-file implementations that solve with LiftCube environment with rgbd or state observations. All scripts contain the same arguments and can be run as so


```
# Training
python sb3_ppo_liftcube_rgbd.py

# Evaluation
python sb3_ppo_liftcube_rgbd.py --eval --model-path=path/to/model
````

Pass in `--help` for more options (e.g. logging, number of parallel environmnets, whether to use ManiSkill2 Vectorized Environments or not etc.)