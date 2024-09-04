# Behavior Cloning

This behavior cloning implementation is adapted from [here](https://github.com/corl-team/CORL/blob/main/algorithms/offline/any_percent_bc.py).

## Installation

To get started, we recommend using conda/mamba to create a new environment and install the dependencies

```shell
conda create -n behavior-cloning-ms python=3.9
conda activate behavior-cloning-ms
pip install -e .
```

## Demonstration Download and Preprocessing

By default for fast downloads and smaller file sizes, ManiSkill demonstrations are stored in a highly reduced/compressed format which includes not keeping any observation data. Run the command to download the demonstration and convert it to a format that includes observation data and the desired action space.

```shell
python -m mani_skill.utils.download_demo "PickCube-v1"
```

```shell
env_id="PickCube-v1"
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/${env_id}/motionplanning/trajectory.h5 \
  --use-first-env-state --allow-failure \
  -c pd_ee_delta_pose -o state \ # -o rgbd for visual observations
  --save-traj --num-procs 10 -b cpu
```

## Training

We provide scripts for state based and rgbd based training. Make sure to use the same sim backend as the backend the demonstrations were collected with. 

Moreover, some demonstrations are slow and can exceed the default max episode steps. In this case, you can use the `--max-episode-steps` flag to set a higher value. Most of the time 2x the default value is sufficient.

For state based training:

```shell
python bc.py --env-id "PickCube-v1" \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pose.cpu.h5 \
  --control-mode "pd_ee_delta_pose" --sim-backend "cpu" --max-episode-steps 100
```

For rgbd based training:

```shell
python bc_rgbd.py --env "PickCube-v1" \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pose.cpu.h5 \
  --control-mode "pd_ee_delta_pose" --sim-backend "cpu" --max-episode-steps 100
```
