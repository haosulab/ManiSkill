# Imitation Learning with ManiSkill2

This contains single-file implementations that solve with LiftCube environment with rgbd or state observations. All scripts contain the same arguments and can be run as so


```
# Training
python bc_liftcube_rgbd.py --demos=demos/rigid_body/LiftCube-v0/trajectory.rgbd.pd_ee_delta_pose.h5

# Evaluation
python bc_liftcube_rgbd.py --eval --model-path=path/to/model
````

Pass in `--help` for more options (e.g. logging, )