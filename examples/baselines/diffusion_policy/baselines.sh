# Baseline scripts

# state based baselines
seed=1
demos=100
for demo_type in "motionplanning" "rl"
do
  # demo_path=~/.maniskill/demos/PickCube-v1/${demo_type}/trajectory.state.pd_ee_delta_pos.cpu.h5
  # if [ -f "$demo_path" ]; then
  #   python train.py --env-id PickCube-v1 \
  #     --demo-path $demo_path \
  #     --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --num-demos ${demos} --max_episode_steps 100 \
  #     --total_iters 30000 \
  #     --exp-name diffusion_policy-PickCube-v1-state-${demos}_motionplanning_demos-${seed} \
  #     --demo_type=${demo_type} --track # additional tag for logging purposes on wandb
  # else
  #   echo "Demo path $demo_path does not exist. Skipping PickCube-v1 for ${demo_type}."
  # fi

  # demo_path=~/.maniskill/demos/PushCube-v1/${demo_type}/trajectory.state.pd_ee_delta_pos.cpu.h5
  # if [ -f "$demo_path" ]; then
  #   python train.py --env-id PushCube-v1 \
  #     --demo-path $demo_path \
  #     --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --num-demos ${demos} --max_episode_steps 100 \
  #     --total_iters 30000 \
  #     --exp-name diffusion_policy-PushCube-v1-state-${demos}_motionplanning_demos-${seed} \
  #     --demo_type=${demo_type} --track # additional tag for logging purposes on wandb
  # else
  #   echo "Demo path $demo_path does not exist. Skipping PushCube-v1 for ${demo_type}."
  # fi
  demo_path=~/.maniskill/demos/AnymalCReach-v1/${demo_type}/trajectory.state.pd_joint_delta_pos.cuda.h5
  if [ -f "$demo_path" ]; then
    python train.py --env-id AnymalCReach-v1 \
      --demo-path $demo_path \
      --control-mode "pd_joint_delta_pos" --sim-backend "gpu" --num-demos ${demos} --max_episode_steps 100 \
      --total_iters 30000 \
      --exp-name diffusion_policy-AnymalCReach-v1-state-${demos}_${demo_type}_demos-${seed} \
      --demo_type=${demo_type} --track # additional tag for logging purposes on wandb
  else
    echo "Demo path $demo_path does not exist. Skipping AnymalCReach-v1 for ${demo_type}."
  fi

  demo_path=~/.maniskill/demos/PushT-v1/${demo_type}/trajectory.state.pd_ee_delta_pos.cpu.h5
  if [ -f "$demo_path" ]; then
    python train.py --env-id PushT-v1 \
      --demo-path $demo_path \
      --control-mode "pd_ee_delta_pos" --sim-backend "gpu" --num-demos ${demos} --max_episode_steps 200 \
      --total_iters 30000 \
      --exp-name diffusion_policy-PushT-v1-state-${demos}_${demo_type}_demos-${seed} \
      --demo_type=${demo_type} --track # additional tag for logging purposes on wandb
  else
    echo "Demo path $demo_path does not exist. Skipping PushT-v1 for ${demo_type}."
  fi

  demo_path=~/.maniskill/demos/StackCube-v1/${demo_type}/trajectory.state.pd_ee_delta_pos.cpu.h5
  if [ -f "$demo_path" ]; then
    python train.py --env-id StackCube-v1 \
      --demo-path $demo_path \
      --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --num-demos ${demos} --max_episode_steps 100 \
      --total_iters 30000 \
      --exp-name diffusion_policy-StackCube-v1-state-${demos}_${demo_type}_demos-${seed} \
      --demo_type=${demo_type} --track # additional tag for logging purposes on wandb
  else
    echo "Demo path $demo_path does not exist. Skipping StackCube-v1 for ${demo_type}."
  fi

  demo_path=~/.maniskill/demos/PegInsertionSide-v1/${demo_type}/trajectory.state.pd_ee_delta_pose.cpu.h5
  if [ -f "$demo_path" ]; then
    python train.py --env-id PegInsertionSide-v1 \
      --demo-path $demo_path \
      --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --num-demos ${demos} --max_episode_steps 100 \
      --total_iters 30000 \
      --exp-name diffusion_policy-PegInsertionSide-v1-state-${demos}_motionplanning_demos-${seed} \
      --demo_type=${demo_type} --track # additional tag for logging purposes on wandb
  else
    echo "Demo path $demo_path does not exist. Skipping PegInsertionSide-v1 for ${demo_type}."
  fi
done
