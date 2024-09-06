### Example scripts for training Diffusion Policy that have some results ###

# Learning from motion planning generated demonstrations

# PushCube-v1
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state \
  --save-traj --num-procs 10 -b cpu

python train.py --env-id PushCube-v1 \
  --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --num-demos 100 --max_episode_steps 100 \
  --total_iters 30000 

# PickCube-v1
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state \
  --save-traj --num-procs 10 -b cpu

python train.py --env-id PickCube-v1 \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --num-demos 100 --max_episode_steps 100 \
  --total_iters 30000 

# StackCube-v1
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state \
  --save-traj --num-procs 10 -b cpu

python train.py --env-id StackCube-v1 \
  --demo-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --num-demos 100 --max_episode_steps 100 \
  --total_iters 30000 

# PegInsertionSide-v1
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pose -o state \
  --save-traj --num-procs 10 -b cpu

python train.py --env-id PegInsertionSide-v1 \
  --demo-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_ee_delta_pose.cpu.h5 \
  --control-mode "pd_ee_delta_pose" --sim-backend "cpu" --num-demos 100 --max_episode_steps 300 \
  --total_iters 300000