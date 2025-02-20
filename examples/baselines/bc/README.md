# Behavior Cloning

This behavior cloning implementation is adapted from [here](https://github.com/corl-team/CORL/blob/main/algorithms/offline/any_percent_bc.py).

## Installation

To get started, we recommend using conda/mamba to create a new environment and install the dependencies

```shell
conda create -n behavior-cloning-ms python=3.9
conda activate behavior-cloning-ms
pip install -e .
```

## Setup

Read through the [imitation learning setup documentation](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html) which details everything you need to know regarding running imitation learning baselines in ManiSkill. It includes details on how to download demonstration datasets, preprocess them, evaluate policies fairly for comparison, as well as suggestions to improve performance and avoid bugs.

## Training

We provide scripts to train Behavior Cloning on demonstrations.

Note that some demonstrations are slow (e.g. motion planning or human teleoperated) and can exceed the default max episode steps which can be an issue as imitation learning algorithms learn to solve the task at the same speed the demonstrations solve it. In this case, you can use the `--max-episode-steps` flag to set a higher value so that the policy can solve the task in time. General recommendation is to set `--max-episode-steps` to about 2x the length of the mean demonstrations length you are using for training. We provide recommended numbers for demonstrations in the examples.sh script.

For state based training:

```shell
python bc.py --env-id "PushCube-v1" \
  --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --max-episode-steps 100 \
  --total-iters 10000
```

<!-- needs to be fixed for some other camera setups
For rgbd based training:

```shell
python bc_rgbd.py --env-id "PushCube-v1" \
  --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --max-episode-steps 100 \
  --total-iters 10000
``` -->
