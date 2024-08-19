# Soft Actor Critic (SAC)

Code for running the PPO RL algorithm is adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl/) and our previous [ManiSkill Baselines](https://github.com/tongzhoumu/ManiSkill_Baselines/). It is written to be single-file and easy to follow/read, and supports state-based RL code.

Note that ManiSkill is still in beta, so we have not finalized training scripts for every pre-built task (some of which are simply too hard to solve with RL anyway). We will further organize these scripts and results in a more organized manner in the future.

## State Based RL

Below is a sample of various commands you can run to train a state-based policy to solve various tasks with SAC that are lightly tuned already. The fastest one is the PushCube-v1 task which can take about 5 minutes to train on a GPU.


```bash
python sac.py --env_id="PushCube-v1" \
  --num_envs=32 --utd=0.5 --buffer_size=200_000 \
  --total_timesteps=200_000 --eval_freq=50_000
python sac.py --env_id="PickCube-v1" \
  --num_envs=32 --utd=0.5 --buffer_size=1_000_000 \
  --total_timesteps=1_000_000 --eval_freq=50_000
```

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