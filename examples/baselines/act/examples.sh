### Example scripts for training ACT that have some results ###

# Learning from motion planning generated demonstrations

# PickCube-v1

# state
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state \
  --save-traj --num-procs 10 -b cpu

python train.py --env-id PickCube-v1 \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --max_episode_steps 100 --total_iters 5000 --save_freq 5000 

# rgbd
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o rgbd \
  --save-traj --num-procs 10 -b cpu
  
python train_rgbd.py --env-id PickCube-v1 \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pos.cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --max_episode_steps 100 --total_iters 20000 --save_freq 20000

# PushCube-v1

# state
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state \
  --save-traj --num-procs 10 -b cpu

python train.py --env-id PushCube-v1 \
  --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --max_episode_steps 100 --total_iters 20000 --save_freq 20000 

# rgbd
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o rgbd \
  --save-traj --num-procs 10 -b cpu

python train.py --env-id PushCube-v1 \
  --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pos.cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --max_episode_steps 100 --total_iters 100000 --save_freq 100000

# StackCube-v1

# state
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state \
  --save-traj --num-procs 10 -b cpu

python train.py --env-id StackCube-v1 \
  --demo-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --max_episode_steps 200 --total_iters 30000 --save_freq 30000

# rgbd
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o rgbd \
  --save-traj --num-procs 10 -b cpu

python train.py --env-id StackCube-v1 \
  --demo-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pos.cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --max_episode_steps 200 --total_iters 100000 --save_freq 100000

# PegInsertionSide-v1

# state
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pose -o state \
  --save-traj --num-procs 10 -b cpu

python train.py --env-id PegInsertionSide-v1 \
  --demo-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_ee_delta_pose.cpu.h5 \
  --control-mode "pd_ee_delta_pose" --sim-backend "cpu" --max_episode_steps 300 --total_iters 300000 --save_freq 300000

# rgbd
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pose -o rgbd \
  --save-traj --num-procs 10 -b cpu

python train.py --env-id PegInsertionSide-v1 \
  --demo-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.rgbd.pd_ee_delta_pose.cpu.h5 \
  --control-mode "pd_ee_delta_pose" --sim-backend "cpu" --max_episode_steps 300 --total_iters 1000000 --save_freq 1000000
