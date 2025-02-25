# Diffusion Policy

Code for running the Diffusion Policy algorithm based on ["Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"](https://arxiv.org/abs/2303.04137v4). It is adapted from the [original code](https://github.com/real-stanford/diffusion_policy).

## Installation

To get started, we recommend using conda/mamba to create a new environment and install the dependencies

```bash
conda create -n diffusion-policy-ms python=3.9
conda activate diffusion-policy-ms
pip install -e .
```

## Setup

Read through the [imitation learning setup documentation](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html) which details everything you need to know regarding running imitation learning baselines in ManiSkill. It includes details on how to download demonstration datasets, preprocess them, evaluate policies fairly for comparison, as well as suggestions to improve performance and avoid bugs.

## Training

We provide scripts to train Diffusion Policy on demonstrations.

Note that some demonstrations are slow (e.g. motion planning or human teleoperated) and can exceed the default max episode steps which can be an issue as imitation learning algorithms learn to solve the task at the same speed the demonstrations solve it. In this case, you can use the `--max-episode-steps` flag to set a higher value so that the policy can solve the task in time. General recommendation is to set `--max-episode-steps` to about 2x the length of the mean demonstrations length you are using for training. We have tuned baselines in the `baselines.sh` script that set a recommended `--max-episode-steps` for each task. Note we have not yet tuned/tested DP for RGB+Depth, just RGB or state only.

Example state based training, learning from 100 demonstrations generated via motionplanning in the PickCube-v1 task

```bash
seed=1
demos=100
python train.py --env-id PickCube-v1 \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 100 \
  --total_iters 30000 \
  --exp-name diffusion_policy-PickCube-v1-state-${demos}_motionplanning_demos-${seed} \
  --track # track training on wandb
```

Example RGB based training (which currently assumes input images are 128x128), learning from 100 demonstrations generated via motionplanning in the PickCube-v1 task

```bash
seed=1
demos=100
python train_rgbd.py --env-id PickCube-v1 \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 100 \
  --total_iters 30000 --obs-mode "rgb" \
  --exp-name diffusion_policy-PickCube-v1-rgb-${demos}_motionplanning_demos-${seed} \
  --track
```

## Citation

If you use this baseline please cite the following
```
@inproceedings{DBLP:conf/rss/ChiFDXCBS23,
  author       = {Cheng Chi and
                  Siyuan Feng and
                  Yilun Du and
                  Zhenjia Xu and
                  Eric Cousineau and
                  Benjamin Burchfiel and
                  Shuran Song},
  editor       = {Kostas E. Bekris and
                  Kris Hauser and
                  Sylvia L. Herbert and
                  Jingjin Yu},
  title        = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  booktitle    = {Robotics: Science and Systems XIX, Daegu, Republic of Korea, July
                  10-14, 2023},
  year         = {2023},
  url          = {https://doi.org/10.15607/RSS.2023.XIX.026},
  doi          = {10.15607/RSS.2023.XIX.026},
  timestamp    = {Mon, 29 Apr 2024 21:28:50 +0200},
  biburl       = {https://dblp.org/rec/conf/rss/ChiFDXCBS23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```