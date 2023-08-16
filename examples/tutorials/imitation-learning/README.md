# Imitation Learning with ManiSkill2

This contains single-file implementations that solve with LiftCube environment with rgbd or state observations.

To download the dataset and preprocess it for training, run the following commands

```
python -m mani_skill2.utils.download_demo "LiftCube-v0" # download base dataset

# preprocess dataset into one that contains rgbd observations and pd_ee_delta_pose control actions
python -m mani_skill2.trajectory.replay_trajectory --traj-path demos/rigid_body/LiftCube-v0/trajectory.h5 \
    --save-traj -o rgbd -c pd_ee_delta_pose --num-procs 8

# preprocess dataset into one that contains state observations and pd_ee_delta_pose control actions
python -m mani_skill2.trajectory.replay_trajectory --traj-path demos/rigid_body/LiftCube-v0/trajectory.h5 \
    --save-traj -o state -c pd_ee_delta_pose --num-procs 8
```

With the data downloaded and processed, you can test Behavior Cloning (BC) with the following commands. All scripts share the same command line arguments.

```
# Training
python bc_liftcube_rgbd.py --demos=demos/rigid_body/LiftCube-v0/trajectory.rgbd.pd_ee_delta_pose.h5

# Evaluation
python bc_liftcube_rgbd.py --eval --model-path=path/to/model
````

Pass in `--help` for more options (e.g. logging, )