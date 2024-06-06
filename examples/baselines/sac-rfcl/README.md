# Reverse Forward Curriculum Learning

Fast offline/online imitation learning in simulation based on "Reverse Forward Curriculum Learning for Extreme Sample and Demo Efficiency in Reinforcement Learning (ICLR 2024)". Code adapted from https://github.com/StoneT2000/rfcl/

Currently this code only works with environments that do not have geometry variations between parallel environments (e.g. PickCube).

Code has been tested and working on the following environments: PickCube-v1

This implementation currently does not include the forward curriculum.

## Download and Process Dataset

Download demonstrations for a desired task e.g. PickCube-v1
```bash
python -m mani_skill.utils.download_demo "PickCube-v1"
```

Process the demonstrations in preparation for the imitation learning workflow
```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/teleop/trajectory.h5 \
  --use-first-env-state -b "gpu" \
  -c pd_joint_delta_pos -o state \
  --save-traj
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/teleop/trajectory.h5 \
  --use-first-env-state -b "gpu" \
  -c pd_joint_delta_pos -o state \
  --save-traj
```

## Train

```bash
python sac_rfcl.py --env_id="PickCube-v1" \
  --num_envs=16 --training_freq=32 --utd=0.5 --buffer_size=1_000_000 \
  --total_timesteps=1_000_000 --eval_freq=25_000 \
  --dataset_path=~/.maniskill/demos/PickCube-v1/teleop/trajectory.state.pd_joint_delta_pos.h5 \
  --num-demos=5 --seed=2 --save_train_video_freq=15
python sac_rfcl.py --env_id="PickCube-v1" \
  --num_envs=16 --training_freq=32 --utd=0.5 --buffer_size=1_000_000 \
  --total_timesteps=1_000_000 --eval_freq=25_000 \
  --dataset_path=../../../demos/PickCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.h5 \
  --num-demos=5 --seed=2 --save_train_video_freq=15 --reverse-step-size=3 --demo_horizon_to_max_steps_ratio=1.5
python sac_rfcl.py --env_id="PickCube-v1" \
  --num_envs=64 --training_freq=64 --utd=0.25 --buffer_size=500_000 \
  --total_timesteps=5_000_000 --eval_freq=25_000 \
  --dataset_path=~/.maniskill/demos/PickCube-v1/teleop/trajectory.state.pd_joint_delta_pos.h5 \
  --num-demos=5 --seed=2 --save_train_video_freq=30
python sac_rfcl.py --env_id="PickCube-v1" \
  --num_envs=128 --training_freq=128 --utd=0.125 --buffer_size=10_000 \
  --total_timesteps=5_000_000 --eval_freq=25_000 \
  --dataset_path=~/.maniskill/demos/PickCube-v1/teleop/trajectory.state.pd_joint_delta_pos.h5 \
  --num-demos=5 --seed=2 --reverse-step-size=3 --demo_horizon_to_max_steps_ratio=1.5 \
  --exp-name="pickcube-mptrajs_5_point_reverse-3_buf-10k_nooffline_-fast128:128-s2" \
  --reverse_curriculum_sampler="point"

python sac_rfcl.py --env_id="StackCube-v1" \
  --num_envs=128 --training_freq=128 --utd=0.125 --buffer_size=10_000 \
  --total_timesteps=5_000_000 --eval_freq=25_000 \
  --dataset_path=~/.maniskill/demos/StackCube-v1/teleop/trajectory.state.pd_joint_delta_pos.h5 \
  --num-demos=5 --seed=2 --reverse-step-size=3 --demo_horizon_to_max_steps_ratio=1.5 \
  --exp-name="stackcube_traj-5_point_buf-10k_nooffline_fast128:128-s2" \
  --reverse_curriculum_sampler="point"

python ppo.py --env_id="StackCube-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=25_000_000
python ppo_rfcl.py --env_id="StackCube-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=50_000_000 \
  --dataset_path=~/.maniskill/demos/StackCube-v1/teleop/trajectory.state.pd_joint_delta_pos.h5 \
  --num-demos=5 --seed=2 --reverse-step-size=3 --demo_horizon_to_max_steps_ratio=3 \
  --exp-name="ppo_rfcl_stackcube" \
  --reverse_curriculum_sampler="point"



python sac_rfcl.py --env_id="PickCube-v1" \
  --num_envs=8 --training_freq=32 --utd=0.5 --buffer_size=1_000_000 \
  --total_timesteps=5_000_000 --eval_freq=25_000 \
  --dataset_path=~/.maniskill/demos/PickCube-v1/teleop/trajectory.state.pd_joint_delta_pos.h5 \
  --num-demos=5 --seed=1014 --reverse-step-size=4 --demo_horizon_to_max_steps_ratio=3 \
  --exp-name="sacrfcl-pickcube-test" \
  --reverse_curriculum_sampler="geometric"
```


## Additional Notes about Implementation

For SAC with RFCL, we always bootstrap on truncated/done.

## Citation

If you use this baseline please cite the following
```
@article{tao2024rfcl,
  title={Reverse Forward Curriculum Learning for Extreme Sample and Demonstration Efficiency in RL},
  author={Tao, Stone and Shukla, Arth and Chan, Tse-kai and Su, Hao},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year={2024}
}
```