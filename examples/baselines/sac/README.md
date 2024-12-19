# Soft Actor Critic (SAC)

Code for running the SAC RL algorithm is adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl/) and our previous [ManiSkill Baselines](https://github.com/tongzhoumu/ManiSkill_Baselines/). It is written to be single-file and easy to follow/read, and supports state-based RL code.

Note that ManiSkill is still in beta, so we have not finalized training scripts for every pre-built task (some of which are simply too hard to solve with RL anyway). We will further organize these scripts and results in a more organized manner in the future.

## State Based RL

Below is a sample of various commands you can run to train a state-based policy to solve various tasks with SAC that are lightly tuned already. Note that control modes can be changed and can be important for improving sample efficiency.


```bash
python sac.py --env_id="PushCube-v1" \
  --num_envs=32 --utd=0.5 --buffer_size=500_000 \
  --total_timesteps=500_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" 
python sac.py --env_id="PickCube-v1" \
  --num_envs=32 --utd=0.5 --buffer_size=500_000 \
  --total_timesteps=500_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" 
```

## Vision Based RL (RGBD)

Below is a sample of various commands for training a image-based policy with SAC that are lightly tuned. You will need to tune the buffer size accordingly as image based observations can take up a lot of memory. The settings below should all take less than 16GB of GPU memory. The examples.sh file has a full list of tested commands for running visual based SAC successfully on many tasks. Change the `--obs_mode` argument to "rgb", "rgb+depth", "depth" to train on RGB or RGBD observations or Depth observations. 

```bash
python sac_rgbd.py --env_id="PickCube-v1" --obs_mode="rgb" \
  --num_envs=32 --utd=0.5 --buffer_size=300_000 \
  --control-mode="pd_ee_delta_pos" --camera_width=64 --camera_height=64 \
  --total_timesteps=1_000_000 --eval_freq=10_000

python sac_rgbd.py --env_id="PickCube-v1" --obs_mode rgb+depth \
  --num_envs=32 --utd=0.5 --buffer_size=300_000 \
  --control-mode="pd_ee_delta_pos" --camera_width=64 --camera_height=64 \
  --total_timesteps=1_000_000 --eval_freq=10_000 
```

### Notes and Optimization Tips

By default, most environments in ManiSkill generate 128x128 images or larger. However some tasks don't need such high resolution images to be solved like PushCube-v1 and PickCube-v1. Lower camera resolutions significantly reduce memory usage and can speed up training time. Currently the SAC rgbd baseline code is written to support 128x128 and 64x64 images only, for other resolutions you need to modify the neural network architecture accordingly.

You can add `--no-include-state` to exclude any state based information from observations. Note however use this with caution as many environements have goal specification information that is part of the state.

## Citation

If you use this baseline please cite the following
```
@inproceedings{DBLP:conf/icml/HaarnojaZAL18,
  author       = {Tuomas Haarnoja and
                  Aurick Zhou and
                  Pieter Abbeel and
                  Sergey Levine},
  editor       = {Jennifer G. Dy and
                  Andreas Krause},
  title        = {Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
                  with a Stochastic Actor},
  booktitle    = {Proceedings of the 35th International Conference on Machine Learning,
                  {ICML} 2018, Stockholmsm{\"{a}}ssan, Stockholm, Sweden, July
                  10-15, 2018},
  series       = {Proceedings of Machine Learning Research},
  volume       = {80},
  pages        = {1856--1865},
  publisher    = {{PMLR}},
  year         = {2018},
  url          = {http://proceedings.mlr.press/v80/haarnoja18b.html},
  timestamp    = {Wed, 03 Apr 2019 18:17:30 +0200},
  biburl       = {https://dblp.org/rec/conf/icml/HaarnojaZAL18.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```