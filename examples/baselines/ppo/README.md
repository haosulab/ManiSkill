# Proximal Policy Optimization (PPO)

Code adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl/)

State based
```bash
python ppo.py --num_envs=1024 --update_epochs=8 --num_minibatches=32 --env_id="PickCube-v1" --total_timesteps=50000000
python ppo.py --num_envs=2048 --update_epochs=8 --num_minibatches=32  --env_id="PushCube-v1" --total_timesteps=100000000 --num-steps=12
python ppo.py --num_envs=1024 --update_epochs=8 --num_minibatches=32 --env_id="StackCube-v1" --total_timesteps=100000000
python ppo.py --num_envs=512 --update_epochs=8 --num_minibatches=32 --env_id="TwoRobotStackCube-v1" --total_timesteps=100000000 --num-steps=100
python ppo.py --num_envs=512 --update_epochs=8 --num_minibatches=32 --env_id="TwoRobotPickCube-v1" --total_timesteps=100000000 --num-steps=100

python ppo.py --num_envs=128 --update_epochs=8 --num_minibatches=32 --env_id="OpenCabinetDrawer-v1" --num-steps=100 --total_timesteps=50000000

python ppo.py --num_envs=1024 --update_epochs=8 --num_minibatches=32 --env_id="QuadrupedRun-v1" --total_timesteps=100000000
```

1024, 100, max100: StackCube-v1__ppo__1__1706348323
1024, 50, max100: StackCube-v1__ppo__1__1706368801
1024, 50, max50: StackCube-v1__ppo__1__1706385843
2048, 25, max25: not a good idea