# Behavior Cloning

This behavior cloning implementation is adapted from [here](https://github.com/corl-team/CORL/blob/main/algorithms/offline/any_percent_bc.py).

## Running the script

1. Install dependencies

```shell
pip install tyro wandb
```

2. Download trajectories for the selected task.

```shell
python -m mani_skill.utils.download_demo "PickCube-v1"
```

3. Replay the trajectories with the correct control mode. The example below performs this on the Pick Cube task with the `pd_ee_delta_pose` control mode and state observations. For the rgbd example change the state to rgbd to record the correct type of observations.

```shell
env_id="PickCube-v1"
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/${env_id}/motionplanning/trajectory.h5 \
  --use-first-env-state --allow-failure \
  -c pd_ee_delta_pose -o state \
  --save-traj --num-procs 4 -b cpu
```

4. Run the script and modify the necessary arguments. A full list of arguments can be found in both of the files.

```shell
python bc.py --env "PickCube-v1" \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pose.cpu.h5 \
  --video --wandb --sim-backend "cpu"
```
