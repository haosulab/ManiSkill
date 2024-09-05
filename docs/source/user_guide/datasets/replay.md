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

```
╭─ options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit                                                               │
│ --traj-path STR         Path to the trajectory .h5 file to replay (required)                                          │
│ --sim-backend STR, -b STR                                                                                             │
│                         Which simulation backend to use. Can be 'auto', 'cpu', 'gpu' (default: auto)                  │
│ --obs-mode {None}|STR, -o {None}|STR                                                                                  │
│                         Target observation mode to record in the trajectory. See                                      │
│                         https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html for a full    │
│                         list of supported observation modes. (default: None)                                          │
│ --target-control-mode {None}|STR, -c {None}|STR                                                                       │
│                         Target control mode to convert the demonstration actions to.                                  │
│                         Note that not all control modes can be converted to others successfully and not all robots    │
│                         have easy to convert control modes.                                                           │
│                         Currently the Panda robots are the best supported when it comes to control mode conversion.   │
│                         (default: None)                                                                               │
│ --verbose, --no-verbose                                                                                               │
│                         Whether to print verbose information during trajectory replays (default: False)               │
│ --save-traj, --no-save-traj                                                                                           │
│                         Whether to save trajectories to disk. This will not override the original trajectory file.    │
│                         (default: False)                                                                              │
│ --save-video, --no-save-video                                                                                         │
│                         Whether to save videos (default: False)                                                       │
│ --num-procs INT         Number of processes to use to help parallelize the trajectory replay process. This uses CPU   │
│                         multiprocessing                                                                               │
│                         and only works with the CPU simulation backend at the moment. (default: 1)                    │
│ --max-retry INT         Maximum number of times to try and replay a trajectory until the task reaches a success state │
│                         at the end. (default: 0)                                                                      │
│ --discard-timeout, --no-discard-timeout                                                                               │
│                         Whether to discard episodes that timeout and are truncated (depends on the max_episode_steps  │
│                         parameter of task) (default: False)                                                           │
│ --allow-failure, --no-allow-failure                                                                                   │
│                         Whether to include episodes that fail in saved videos and trajectory data (default: False)    │
│ --vis, --no-vis         Whether to visualize the trajectory replay via the GUI. (default: False)                      │
│ --use-env-states, --no-use-env-states                                                                                 │
│                         Whether to replay by environment states instead of actions. This guarantees that the          │
│                         environment will look exactly                                                                 │
│                         the same as the original trajectory at every step. (default: False)                           │
│ --use-first-env-state, --no-use-first-env-state                                                                       │
│                         Use the first env state in the trajectory to set initial state. This can be useful for trying │
│                         to replay                                                                                     │
│                         demonstrations collected in the CPU simulation in the GPU simulation by first starting with   │
│                         the same initial                                                                              │
│                         state as GPU simulated tasks will randomize initial states differently despite given the same │
│                         seed compared to CPU sim. (default: False)                                                    │
│ --count {None}|INT      Number of demonstrations to replay before exiting. By default will replay all demonstrations  │
│                         (default: None)                                                                               │
│ --reward-mode {None}|STR                                                                                              │
│                         Specifies the reward type that the env should use. By default it will pick the first          │
│                         supported reward mode. Most environments                                                      │
│                         support 'sparse', 'none', and some further support 'normalized_dense' and 'dense' reward      │
│                         modes (default: None)                                                                         │
│ --record-rewards, --no-record-rewards                                                                                 │
│                         Whether the replayed trajectory should include rewards (default: False)                       │
│ --shader STR            Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt'   │
│                         for ray tracing                                                                               │
│                         and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality  │
│                         ray-traced renderer (default: default)                                                        │
│ --video-fps INT         The FPS of saved videos (default: 30)                                                         │
│ --render-mode STR       The render mode used for saving videos. Typically there is also 'sensors' and 'all' render    │
│                         modes which further render all sensor outputs like cameras. (default: rgb_array)              │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
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

### Replaying Trajectories from One Control Mode to a Easier to Learn Control Mode

In machine learning workflows, it can sometimes be easier to learn from some control modes such as end-effector control ones. The example below does that exactly

```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path path/to/trajectory.h5 \
  -c pd_ee_delta_pose -o state \
  --save-traj
```

Note that some target control modes are difficult to convert to due to inherent differences in controllers. For highly precise tasks like PegInsertionSide and PlugCharger the success rate of conversion from e.g. `pd_joint_pos` control to `pd_ee_delta_pose` control is low. For less precise tasks like PickCube the success rate is near 100%.

### Adding rewards/observations in trajectories

To conserve memory, demonstrations are stored without observations and rewards. The example below shows how to add rewards and RGB observations back in.

```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path path/to/trajectory.h5 \
  --record-rewards --reward-mode="normalized_dense" -o rgb \
  --save-traj
```
