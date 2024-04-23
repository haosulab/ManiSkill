# Replaying/Converting Trajectories

ManiSkill provides tools to not only collect/load trajectories, but to also replay trajectories and convert observations/actions.

To replay the demonstrations (without changing the observation mode and control mode):

```bash
# Replay and view trajectories through sapien viewer
python -m mani_skill.trajectory.replay_trajectory --traj-path demos/rigid_body/PickCube-v1/trajectory.h5 --vis

# Save videos of trajectories (to the same directory of trajectory)
python -m mani_skill.trajectory.replay_trajectory --traj-path demos/rigid_body/PickCube-v1/trajectory.h5 --save-video

# see a full list of options
python -m mani_skill.trajectory.replay_trajectory -h
```

:::{note}
The script requires `trajectory.h5` and `trajectory.json` to be both under the same directory.
:::

The raw demonstration files contain all the necessary information (e.g. initial states, actions, seeds) to reproduce a trajectory. Observations are not included since they can lead to large file sizes without postprocessing. In addition, actions in these files do not cover all control modes. Therefore, you need to convert the raw files into your desired observation and control modes. We provide a utility script that works as follows:

```bash
# Replay demonstrations with control_mode=pd_joint_delta_pos
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path demos/rigid_body/PickCube-v1/trajectory.h5 \
  --save-traj --target-control-mode pd_joint_delta_pos --obs-mode none --num-procs 10
```

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

<br>

:::{note}
For soft-body tasks, please compile and generate caches (`python -m mani_skill.utils.precompile_mpm`) before running the script with multiple processes (with `--num-procs`).
:::

:::{caution}
The conversion between controllers (or action spaces) is not yet supported for mobile manipulators (e.g., used in tasks migrated from ManiSkill1).
:::

:::{caution}
Since some demonstrations are collected in a non-quasi-static way (objects are not fixed relative to the manipulator during manipulation) for some challenging tasks (e.g., `TurnFaucet` and tasks migrated from ManiSkill1), replaying actions can fail due to non-determinism in simulation. Thus, replaying trajectories by environment states is required (passing `--use-env-states`).
:::

---

We recommend using our script only for converting actions into different control modes without recording any observation information (i.e. passing `--obs-mode=none`). The reason is that (1) some observation modes, e.g. point cloud, can take too much space without any post-processing, e.g., point cloud downsampling; in addition, the `state` mode for soft-body tasks also has a similar issue, since the states of those tasks are particles. (2) Some algorithms  (e.g. GAIL) require custom keys stored in the demonstration files, e.g. next-observation.

Thus we recommend that, after you convert actions into different control modes, implement your custom environment wrappers for observation processing. After this, use another script to render and save the corresponding post-processed visual demonstrations. [ManiSkill-Learn](https://github.com/haosulab/ManiSkill-Learn) has included such observation processing wrappers and demonstration conversion script (with multi-processing), so we recommend referring to the repo for more details.
