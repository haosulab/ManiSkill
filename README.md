# ManiSkill2

![teaser](figures/teaser.jpg)

ManiSkill2 is a large-scale robotic manipulation benchmark, focusing on learning generalizable robot agents and manipulation skills. It features 2000+ diverse objects, 20 task categories, and a large-scale demonstration set in [SAPIEN](https://sapien.ucsd.edu/), a fully-physical, realistic simulator. The benchmark can be used to study 2D & 3D vision-based imitation learning, reinforcement learning, and motion planning, etc. We invite you to participate in the associated [ManiSkill 2022 challenge](https://sapien.ucsd.edu/challenges/maniskill/2022/) where we will be awarding prizes to the teams who achieve the highest success rates in our environments.

**Notes**: We are actively introducing new functionalities and improvements (e.g. new tasks and highly efficient system for visual RL). See the [roadmap](https://github.com/haosulab/ManiSkill2/discussions/30) for more details.

**Table of Contents**

- [Installation](#installation)
- [Getting Started](#getting-started)
  - [Interactive play](#interactive-play)
  - [Environment Interface](#environment-interface)
- [Reinforcement Learning Example with ManiSkill2-Learn](#reinforcement-learning-example-with-maniskill2-learn)
- [Demonstrations](#demonstrations)
- [ManiSkill 2022 Challenge](#maniskill-2022-challenge)
- [Leaderboard](#leaderboard)
- [License](#license)

## Installation

First, clone the repo:

```bash
git clone https://github.com/haosulab/ManiSkill2.git
```

Then, install dependencies and this package `mani_skill2`:

```bash
conda env create -n mani_skill2 -f environment.yml
conda activate mani_skill2
python setup.py develop
```

`gym>0.21` introduces breaking changes, e.g., deprecating `env.seed()`. We recommend `pip install gym==0.18.3 --no-deps`.

Some environments require **downloading assets**. You can download all the assets by `python tools/download.py --uid all`.

---

> The following section is to install Warp for soft-body environments. Skip if you do not need it.

To run soft body environments, **CUDA toolkit >= 11.3 and gcc** are required.
You can download and install the CUDA toolkit from
<https://developer.nvidia.com/cuda-downloads?target_os=Linux>.
Assuming the CUDA toolkit is installed at `/usr/local/cuda`, you need to ensure `CUDA_PATH` or `CUDA_HOME` is set properly:

```bash
export CUDA_PATH=/usr/local/cuda

# The following command should print a CUDA compiler version >= 11.3
${CUDA_PATH}/bin/nvcc --version

# The following command should output a valid gcc version
gcc --version
```

If `nvcc` is included in `$PATH`, we will try to figure out the variable `CUDA_PATH` automatically.

To verify CUDA is properly set up for ManiSkill2, run the following in the root directory of this repository to compile warp.

``` bash
python warp_maniskill/build_lib.py
```

For soft body environments, you need to make sure only 1 CUDA device is visible:

``` bash
# Select the first CUDA device. Change 0 to other integer for other device.
export CUDA_VISIBLE_DEVICES=0
```

If multiple CUDA devices are visible, the environment will give an error. If you
want to interactively visualize the environment, you need to assign the id of
the GPU connected to your display (e.g., monitor screen).

All soft body environments require runtime compilation and cache generation. You
can run the following to compile and generate cache in advance. **This step is
required if you run soft body environments in parallel with multiple processes.**

``` bash
python tools/precompile_mpm.py
```

## Getting Started

### Interactive play

We provide a demo script to interactively play with our environments.

```bash
python examples/demo_manual_control.py -e PickCube-v0
# PickCube-v0 can be replaced with other environment id.
```

Press `i` (or `j`, `k`, `l`, `u`, `o`) to move the end-effector. Press any key between `1` to `6` to rotate the end-effector. Press `f` or `g` to open or close the gripper. Press `q` to close the viewer and exit the program.

For `PickCube-v0`, the green sphere indicates the goal position to move the cube to. See our wiki pages for more [rigid-body environments](https://github.com/haosulab/ManiSkill2/wiki/Rigid-Body-Environments) and [soft-body environments](https://github.com/haosulab/ManiSkill2/wiki/Soft-Body-Environments). You can also download assets individually for certain environments (e.g. `PickSingleYCB-v0`, `TurnFaucet-v0`, `AssemblingKits-v0`)  following the above wiki pages.

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

The supported observation modes are `pointcloud`, `rgbd`, `state_dict` and `state`. Note that for the Maniskill 2022 Challenge, only `pointcloud` and `rgbd` are permitted.

Please refer to our wiki for information on the [observation](https://github.com/haosulab/ManiSkill2/wiki/Observation-Space) and [control](https://github.com/haosulab/ManiSkill2/wiki/Controllers) modes available and their details.

## Reinforcement Learning Example with ManiSkill2-Learn

We provide [ManiSkill2-Learn](https://github.com/haosulab/ManiSkill2-Learn), an improved framework based on [ManiSkill-Learn](https://github.com/haosulab/ManiSkill-Learn) for training RL agents with demonstrations to solve manipulation tasks. The framework conveniently supports both point cloud-based and RGB-D-based policy learning, and the custom processing of these visual observations. It also supports many common algorithms (BC, PPO, DAPG, SAC, GAIL). Moreover, this framework is optimized for point cloud-based policy learning, and includes some helpful and empirical advice to get you started.

## Demonstrations

We provide a dataset of expert demonstrations to facilitate learning-from-demonstrations approaches, e.g., [Shen et. al.](https://arxiv.org/pdf/2203.02107.pdf).
The datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1hVdUNPGCHh0OULPCowBClPYIXSwsx-J9).
For those who cannot access Google Drive, the datasets can be downloaded from [ScienceDB.cn](http://doi.org/10.57760/sciencedb.02239).

To bulk download demonstrations, you can use the following scripts:

```bash
pip install gdown

# Download all rigid-body demonstrations
gdown https://drive.google.com/drive/folders/1pd9Njg2sOR1VSSmp-c1mT7zCgJEnF8r7 --folder -O demos/

# Download all soft-body demonstrations
gdown https://drive.google.com/drive/folders/1QCYgcmRs9SDhXj6fVWPzuv7ZSBL94q2R --folder -O demos/

# Download task-specific demonstrations
gdown ${TASK_SPECIFIC_FOLDER_URL} --folder

# Download individual demonstrations
gdown ${DEMO_URL}
```

All demonstrations for an environment are saved in the HDF5 format and stored in their corresponding folders on [Google Drive](https://drive.google.com/drive/folders/1hVdUNPGCHh0OULPCowBClPYIXSwsx-J9). Each dataset name is formatted as `trajectory.{obs_mode}.{control_mode}.h5`. Each dataset is associated with a JSON file with the same base name. In each folder, `trajectory.h5` contains the original demonstrations generated by the `pd_joint_pos` controller. See the [wiki page on demonstrations](https://github.com/haosulab/ManiSkill2/wiki/Demonstrations) for details, such as formats.

To replay the demonstrations (without changing the observation mode and control mode):

```bash
# Replay and view trajectories through sapien viewer
python tools/replay_trajectory.py --traj-path demos/rigid_body_envs/PickCube-v0/trajectory.h5 --vis

# Save videos of trajectories 
python tools/replay_trajectory.py --traj-path demos/rigid_body_envs/PickCube-v0/trajectory.h5 --save-video
```

> The script requires `trajectory.h5` and `trajectory.json` to be both under the same directory

The raw demonstration files contain all necessary information (e.g. initial states, actions, seeds) to reproduce a trajectory. Observations are not included since they can lead to large file sizes without postprocessing. In addition, actions in these files do not cover all control modes. Therefore, you need to convert our raw files into your desired observation and control modes. We provide a utility script that works as follows:

```bash
# Replay demonstrations with control_mode=pd_joint_delta_pos
python tools/replay_trajectory.py --traj-path demos/rigid_body_envs/PickCube-v0/trajectory.h5 \
  --save-traj --target-control-mode pd_joint_delta_pos --obs-mode none --num-procs 10
```

> For soft-body environments, please compile and generate caches (`python tools/precompile_mpm.py`) before running the script with multiple processes (with `--num-procs`).

<details>
  
<summary><b>Click here</b> for important notes about the script arguments.</summary>

- `--save-traj`: save the replayed trajectory to the same folder as the original trajectory file.
- `--num-procs=10`: split trajectories to multiple processes (e.g., 10 processes) for acceleration.
- `--obs-mode=none`: specify the observation mode as `none`, i.e. not saving any observations.
- `--obs-mode=rgbd`: (not included in the script above) specify the observation mode as `rgbd` to replay the trajectory. If `--save-traj`, the saved trajectory will contain the RGBD observations. RGB images are saved as uint8 and depth images (multiplied by 1024) are saved as uint16.
- `--obs-mode=pointcloud`: (not included in the script above) specify the observation mode as `pointcloud`. We encourage you to further process the point cloud instead of using this point clould directly.
- `--obs-mode=state`: (not included in the script above) specify the observation mode as `state`. Note that the `state` observation mode is not allowed for challenge submission.
  
</details>

We recommend using our script only for converting actions into different control modes without recording any observation information (i.e. passing `--obs-mode=none`). The reason is that (1) some observation modes, e.g. point cloud, can take much space without any post-processing, e.g., point cloud downsampling; in addition, the `state` mode for soft-body environments also has a similar issue, since the states of those environments are particles. (2) Some algorithms  (e.g. GAIL) require custom keys stored in the demonstration files, e.g. next-observation.
  
Thus we recommend that, after you convert actions into different control modes, implement your custom environment wrappers for observation processing. After this, use another script to render and save the corresponding post-processed visual demonstrations. [ManiSkill2-Learn](https://github.com/haosulab/ManiSkill2-Learn) has included such observation processing wrapper and demonstration conversion script (with multi-processing), so we recommend referring to the repo for more details.

## ManiSkill 2022 Challenge

The ManiSkill 2022 challenge is an ongoing competition using the ManiSkill2 benchmark. See our [website](https://sapien.ucsd.edu/challenges/maniskill/2022/) for additional competition details and follow the [getting started](https://sapien.ucsd.edu/challenges/maniskill/2022#getting-started) section to learn how to compete.

To create a submission for the competition, follow [the instructions on our wiki](https://github.com/haosulab/ManiSkill2/wiki/Participation-Guidelines) on how to create a submission and submit it to the leaderboard.

Previous results of the ManiSkill 2021 challenge can be found [here](https://sapien.ucsd.edu/challenges/maniskill/2022/#maniskill2021). Winning solutions and their codes can be found in the previous challenge.

## Leaderboard

You can find the leaderboard on the challenge website: <https://sapien.ucsd.edu/challenges/maniskill/challenges/ms2022>.

## License

All rigid body environments in ManiSkill are licensed under fully permissive licenses (e.g., Apache-2.0).

However, the soft body environments will follow Warp's license. Currently, they are licensed under
[NVIDIA Source Code License for Warp](https://github.com/NVIDIA/warp/blob/main/LICENSE.md).

The assets are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode).
