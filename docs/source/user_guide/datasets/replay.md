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

By default raw demonstration files contain all the necessary information (e.g. initial states, actions, seeds) to reproduce a trajectory. Observations are not included since they can lead to large file sizes without postprocessing. In addition, actions in these files do not cover all control modes. Therefore, you need to convert the raw files into your desired observation and control modes. We provide a utility script that works as follows:

```bash
# Replay demonstrations with control_mode=pd_joint_delta_pos
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path demos/rigid_body/PickCube-v1/trajectory.h5 \
  --save-traj --target-control-mode pd_joint_delta_pos \
  --obs-mode none --num-procs 10
```


:::{dropdown} Click here to see the replay trajectory tool options

Command Line Options:

- `--save-traj`: save the replayed trajectory to the same folder as the original trajectory file.
- `--target-control-mode`: The target control mode / action space to save into the trajectory file.
- `--save-video`: Whether to save a video of the replayed trajectories
- `--max-retry`: Max number of times to try and replay each trajectory
- `--discard-timeout`: Whether to discard trajectories that time out due to the default environment's max episode steps config
- `--allow-failure`: Whether to permit saving failed trajectories
- `--vis`: Whether to open the GUI and show the replayed trajectories on a display
- `--use-first-env-state`: Whether to use the first environment state of the given trajectory to initialize the environment
- `--num-procs=10`: split trajectories to multiple processes (e.g., 10 processes) for acceleration. Note this is done via CPU parallelization, not GPU. This argument is also currently incompatible with using the GPU simulation to replay trajectories.
- `--obs-mode=none`: specify the observation mode as `none`, i.e. not saving any observations.
- `--obs-mode=rgbd`: (not included in the script above) specify the observation mode as `rgbd` to replay the trajectory. If `--save-traj`, the saved trajectory will contain the RGBD observations.
- `--obs-mode=pointcloud`: (not included in the script above) specify the observation mode as `pointcloud`. We encourage you to further process the point cloud instead of using this point cloud directly (e.g. sub-sampling the pointcloud)
- `--obs-mode=state`: (not included in the script above) specify the observation mode as `state`
- `--use-env-states`: For each time step $t$, after replaying the action at this time step and obtaining a new observation at $t+1$, set the environment state at time $t+1$ as the recorded environment state at time $t+1$. This is necessary for successfully replaying trajectories for the tasks migrated from ManiSkill1.
- `--count`: Number of demonstrations to replay before exiting. By default all demonstrations are replayed
- `--shader`: "Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"
- `--render-mode`: The render mode used in the video saving
- `-b, --sim-backend`: Which simulation backend to use. Can be 'auto', 'cpu', or 'gpu'
:::
<!-- 
:::{note}
For soft-body tasks, please compile and generate caches (`python -m mani_skill.utils.precompile_mpm`) before running the script with multiple processes (with `--num-procs`).
::: -->
<!-- 
:::{caution}
The conversion between controllers (or action spaces) is not yet supported for mobile manipulators (e.g., used in tasks migrated from ManiSkill1).
::: -->

:::{caution}
Since some demonstrations are collected in a non-quasi-static way (objects are not fixed relative to the manipulator during manipulation) for some challenging tasks (e.g., `TurnFaucet` and tasks migrated from ManiSkill1), replaying actions can fail due to non-determinism in simulation. Thus, replaying trajectories by environment states is required (passing `--use-env-states`).
:::

## Example Usages

As the replay trajectory tool is fairly complex and feature rich, we suggest a few example workflows that may be useful for various use cases


### Replaying Trajectories collected in CPU/GPU sim to GPU/CPU sim

Some demonstrations may have been collected on the CPU simulation but you want data that works for the GPU simulation and vice versa. Inherently CPU and GPU simulation will have slightly different behaviors given the same actions and the same start state.

For example if you use teleoperation to collect demos, these are often collected in the CPU sim for flexibility and single-thread speed. However imitation/reinforcement learning workflows might use the GPU simulation for training. In order to ensure the demos can be learned from, we can replay them in the GPU simulation and save the ones that replay successfully. This is done by using the first environment state, force using the GPU simulation with `-b "gpu"`, and setting desired control and observation modes.

```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path path/to/trajectory.h5 \
  --use-first-env-state -b "gpu" \
  -c pd_joint_delta_pos -o state \
  --save-traj
```
