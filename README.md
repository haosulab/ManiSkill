# ManiSkill2

![teaser](figures/teaser_v2.jpg)

[![PyPI version](https://badge.fury.io/py/mani-skill2.svg)](https://badge.fury.io/py/mani-skill2)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haosulab/ManiSkill2/blob/main/examples/tutorials/1_quickstart.ipynb)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://haosulab.github.io/ManiSkill2)
<!-- [![Docs](https://github.com/haosulab/ManiSkill2/actions/workflows/gh-pages.yml/badge.svg)](https://haosulab.github.io/ManiSkill2) -->

ManiSkill2 is a unified benchmark for learning generalizable robotic manipulation skills powered by [SAPIEN](https://sapien.ucsd.edu/). **It features 20 out-of-box task families with 2000+ diverse object models and 4M+ demonstration frames**. Moreover, it empowers fast visual input learning algorithms so that **a CNN-based policy can collect samples at about 2000 FPS with 1 GPU and 16 processes on a workstation**. The benchmark can be used to study a wide range of algorithms: 2D & 3D vision-based reinforcement learning, imitation learning, sense-plan-act, etc.

Please refer our [documentation](https://haosulab.github.io/ManiSkill2) to learn more information. There are also hands-on [tutorials](examples/tutorials) (e.g, [quickstart colab tutorial](https://colab.research.google.com/github/haosulab/ManiSkill2/blob/main/examples/tutorials/1_quickstart.ipynb)).

We invite you to participate in the associated [ManiSkill2 challenge](https://sapien.ucsd.edu/challenges/maniskill/) where the top teams will be awarded prizes.

**Table of Contents**

- [Installation](#installation)
- [Getting Started](#getting-started)
  - [Interactive play](#interactive-play)
  - [Environment Interface](#environment-interface)
- [Reinforcement Learning Example with ManiSkill2-Learn](#reinforcement-learning-example-with-maniskill2-learn)
- [Demonstrations](#demonstrations)
- [ManiSkill2 Challenge](#maniskill2-challenge)
- [Leaderboard](#leaderboard)
- [License](#license)
- [Citation](#citation)

## Installation

From pip:

```bash
pip install mani-skill2
```

From github:

```bash
pip install --upgrade git+https://github.com/haosulab/ManiSkill2.git
```

From source:

```bash
git clone https://github.com/haosulab/ManiSkill2.git
cd ManiSkill2 && pip install -e .
```

---

GPU is required to enable rendering for ManiSkill2. The rigid-body environments, powered by SAPIEN, are ready to use after installation.

Test your installation:

```bash
# Run an episode (at most 200 steps) of "PickCube-v0" (a rigid-body environment) with random actions
python -m mani_skill2.examples.demo_random_action
```

Some environments require **downloading assets**. You can download all the assets by `python -m mani_skill2.utils.download_asset all` or download task-specific assets by `python -m mani_skill2.utils.download_asset ${ENV_ID}`.

---

The soft-body environments are based on SAPIEN and customized [NVIDIA Warp](https://github.com/NVIDIA/warp), which requires **CUDA toolkit >= 11.3 and gcc** to compile. Please refer to the [documentation](https://haosulab.github.io/ManiSkill2/getting_started/installation.html#warp-maniskill2-version) for more details about installing ManiSkill2 Warp.

For soft-body environments, you need to make sure only 1 CUDA device is visible:

``` bash
# Select the first CUDA device. Change 0 to other integer for other device.
export CUDA_VISIBLE_DEVICES=0
```

If multiple CUDA devices are visible, the environment will give an error. If you
want to interactively visualize the environment, you need to assign the id of
the GPU connected to your display (e.g., monitor screen).

All soft-body environments require runtime compilation and cache generation. You
can run the following to compile and generate cache in advance. **This step is
required before you run soft body environments in parallel with multiple processes.**

``` bash
python -m mani_skill2.utils.precompile_mpm
```

## Getting Started

### Interactive play

We provide a demo script to interactively play with our environments.

```bash
# PickCube-v0 can be replaced with other environment id.
python -m mani_skill2.examples.demo_manual_control -e PickCube-v0
```

Keyboard controls:

- Press `i` (or `j`, `k`, `l`, `u`, `o`) to move the end-effector.
- Press any key between `1` to `6` to rotate the end-effector.
- Press `f` or `g` to open or close the gripper.
- Press `w` (or `a`, `s`, `d`) to translate the base if the robot is mobile. Press `q` or `e` to rotate the base. Press `z` or `x` to lift the torso.
- Press `esc` to close the viewer and exit the program.

For `PickCube-v0`, the green sphere indicates the goal position to move the cube to. Please refer to [Environments](https://haosulab.github.io/ManiSkill2/concepts/environments.html) for all supported environments and whether they require downloading assets.


### Environment Interface

Here is a basic example of how to make an [OpenAI Gym](https://github.com/openai/gym) environment and run a random policy.

```python
import gym
import mani_skill2.envs

env = gym.make("PickCube-v0", obs_mode="rgbd", control_mode="pd_joint_delta_pos")
print("Observation space", env.observation_space)
print("Action space", env.action_space)

env.seed(0)  # specify a seed for randomness
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()  # a display is required to render
env.close()
```

Each `mani_skill2` environment supports different **observation modes** and **control modes**, which determine the **observation space** and **action space**. They can be specified by `gym.make(env_id, obs_mode=..., control_mode=...)`.

The basic observation modes supported are `pointcloud`, `rgbd`, `state_dict` and `state`. Additional observation modes which include robot segmentation masks are `pointcloud+robot_seg` and `rgbd+robot_seg`. Note that for the Maniskill2 Challenge, only `pointcloud` and `rgbd` (and their `robot_seg` versions) are permitted.

Please refer to our documentation for information on the [observation](https://haosulab.github.io/ManiSkill2/concepts/observation.html) and [control](https://haosulab.github.io/ManiSkill2/concepts/controllers.html) modes available and their details.

## Reinforcement Learning Example with ManiSkill2-Learn

We provide [ManiSkill2-Learn](https://github.com/haosulab/ManiSkill2-Learn), an improved framework based on [ManiSkill-Learn](https://github.com/haosulab/ManiSkill-Learn) for training RL agents with demonstrations to solve manipulation tasks. The framework conveniently supports both point cloud-based and RGB-D-based policy learning, and the custom processing of these visual observations. It also supports many common algorithms (BC, PPO, DAPG, SAC, GAIL). Moreover, this framework is optimized for point cloud-based policy learning, and includes some helpful and empirical advice to get you started.

## Demonstrations

We provide a dataset of expert demonstrations to facilitate learning-from-demonstrations approaches, e.g., [Shen et. al.](https://arxiv.org/pdf/2203.02107.pdf).
The datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1hVdUNPGCHh0OULPCowBClPYIXSwsx-J9) or via command line as shown below.

For those who cannot access Google Drive, the datasets can be downloaded from [ScienceDB.cn](http://doi.org/10.57760/sciencedb.02239).

To bulk download demonstrations, you can use the following scripts:

```bash
# Download all rigid-body demonstrations
python -m mani_skill2.utils.download_demo rigid_body -o demos

# Download all soft-body demonstrations
python -m mani_skill2.utils.download_demo soft_body -o demos

# Download task-specific demonstrations
python -m mani_skill2.utils.download_demo PickCube-v0
```

All demonstrations for an environment are saved in the HDF5 format and stored in their corresponding folders on [Google Drive](https://drive.google.com/drive/folders/1hVdUNPGCHh0OULPCowBClPYIXSwsx-J9). Each dataset name is formatted as `trajectory.{obs_mode}.{control_mode}.h5`. Each dataset is associated with a JSON file with the same base name. In each folder, `trajectory.h5` contains the original demonstrations generated by the `pd_joint_pos` controller. See the [wiki page on demonstrations](https://github.com/haosulab/ManiSkill2/wiki/Demonstrations) for details, such as formats.

To replay the demonstrations (without changing the observation mode and control mode):

```bash
# Replay and view trajectories through sapien viewer
python -m mani_skill2.trajectory.replay_trajectory --traj-path demos/rigid_body/PickCube-v0/trajectory.h5 --vis

# Save videos of trajectories 
python -m mani_skill2.trajectory.replay_trajectory --traj-path demos/rigid_body/PickCube-v0/trajectory.h5 --save-video
```

> The script requires `trajectory.h5` and `trajectory.json` to be both under the same directory

The raw demonstration files contain all necessary information (e.g. initial states, actions, seeds) to reproduce a trajectory. Observations are not included since they can lead to large file sizes without postprocessing. In addition, actions in these files do not cover all control modes. Therefore, you need to convert our raw files into your desired observation and control modes. We provide a utility script that works as follows:

```bash
# Replay demonstrations with control_mode=pd_joint_delta_pos
python -m mani_skill2.trajectory.replay_trajectory --traj-path demos/rigid_body/PickCube-v0/trajectory.h5 \
  --save-traj --target-control-mode pd_joint_delta_pos --obs-mode none --num-procs 10
```

> For soft-body environments, please compile and generate caches (`python tools/precompile_mpm.py`) before running the script with multiple processes (with `--num-procs`).

<details>
  
<summary><b>Click here</b> for important notes about the script arguments.</summary>

- `--save-traj`: save the replayed trajectory to the same folder as the original trajectory file.
- `--num-procs=10`: split trajectories to multiple processes (e.g., 10 processes) for acceleration.
- `--obs-mode=none`: specify the observation mode as `none`, i.e. not saving any observations.
- `--obs-mode=rgbd`: (not included in the script above) specify the observation mode as `rgbd` to replay the trajectory. If `--save-traj`, the saved trajectory will contain the RGBD observations. RGB images are saved as uint8 and depth images (multiplied by 1024) are saved as uint16.
- `--obs-mode=pointcloud`: (not included in the script above) specify the observation mode as `pointcloud`. We encourage you to further process the point cloud instead of using this point cloud directly.
- `--obs-mode=state`: (not included in the script above) specify the observation mode as `state`. Note that the `state` observation mode is not allowed for challenge submission.
- `--use-env-states`: For each time step $t$, after replaying the action at this time step and obtaining a new observation at $t+1$, set the environment state at time $t+1$ as the recorded environment state at time $t+1$. This is necessary for successfully replaying trajectories for the tasks migrated from ManiSkill1.
  
</details>

We recommend using our script only for converting actions into different control modes without recording any observation information (i.e. passing `--obs-mode=none`). The reason is that (1) some observation modes, e.g. point cloud, can take much space without any post-processing, e.g., point cloud downsampling; in addition, the `state` mode for soft-body environments also has a similar issue, since the states of those environments are particles. (2) Some algorithms  (e.g. GAIL) require custom keys stored in the demonstration files, e.g. next-observation.
  
Thus we recommend that, after you convert actions into different control modes, implement your custom environment wrappers for observation processing. After this, use another script to render and save the corresponding post-processed visual demonstrations. [ManiSkill2-Learn](https://github.com/haosulab/ManiSkill2-Learn) has included such observation processing wrapper and demonstration conversion script (with multi-processing), so we recommend referring to the repo for more details.

## ManiSkill2 Challenge

The ManiSkill2 challenge is an ongoing competition using the ManiSkill2 benchmark. See our [website](https://sapien.ucsd.edu/challenges/maniskill/) for additional competition details and follow the [getting started](https://sapien.ucsd.edu/challenges/maniskill#getting-started) section to learn how to compete.

To create a submission for the competition, follow [the instructions on our wiki](https://github.com/haosulab/ManiSkill2/wiki/Participation-Guidelines) on how to create a submission and submit it to the leaderboard.

Previous results of the ManiSkill 2021 challenge can be found [here](https://sapien.ucsd.edu/challenges/maniskill#maniskill2021). Winning solutions and their codes can be found in the previous challenge.

## Leaderboard

You can find the leaderboard on the challenge website: <https://sapien.ucsd.edu/challenges/maniskill/challenges/ms2>.

## License

All rigid body environments in ManiSkill are licensed under fully permissive licenses (e.g., Apache-2.0).

However, the soft body environments will follow Warp's license. Currently, they are licensed under
[NVIDIA Source Code License for Warp](https://github.com/NVIDIA/warp/blob/main/LICENSE.md).

The assets are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

## Citation

If you use ManiSkill2 or its assets and models, consider citing the following publication:

```
@inproceedings{gu2023maniskill2,
  title={ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills},
  author={Gu, Jiayuan and Xiang, Fanbo and Li, Xuanlin and Ling, Zhan and Liu, Xiqiaing and Mu, Tongzhou and Tang, Yihe and Tao, Stone and Wei, Xinyue and Yao, Yunchao and Yuan, Xiaodi and Xie, Pengwei and Huang, Zhiao and Chen, Rui and Su, Hao},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
