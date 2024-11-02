### Example scripts for training ACT that have some results ###

# Learning from motion planning generated demonstrations

# PushCube-v1
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state \
  --save-traj --num-procs 10 -b cpu

# PickCube-v1
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state \
  --save-traj --num-procs 10 -b cpu

# StackCube-v1
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state \
  --save-traj --num-procs 10 -b cpu

# PegInsertionSide-v1
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pose -o state \
  --save-traj --num-procs 10 -b cpu
