# Baseline scripts

for demo_type in "motionplanning" "rl"
do
  demo_path=~/.maniskill/demos/PickCube-v1/${demo_type}/trajectory.state.pd_ee_delta_pos.cpu.h5
  if [ -f "$demo_path" ]; then
    python train.py --env-id PickCube-v1 \
      --demo-path $demo_path \
      --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --num-demos 100 --max_episode_steps 100 \
      --total_iters 30000 \
      --exp-name diffusion_policy-${env_id}-state-${demos}_motionplanning_demos-${seed} \
      --demo_type=${demo_type} --track # additional tag for logging purposes on wandb
  else
    echo "Demo path $demo_path does not exist. Skipping PickCube-v1 for ${demo_type}."
  fi

  demo_path=~/.maniskill/demos/PushCube-v1/${demo_type}/trajectory.state.pd_ee_delta_pos.cpu.h5
  if [ -f "$demo_path" ]; then
    python train.py --env-id PushCube-v1 \
      --demo-path $demo_path \
      --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --num-demos 100 --max_episode_steps 100 \
      --total_iters 30000 \
      --exp-name diffusion_policy-${env_id}-state-${demos}_motionplanning_demos-${seed} \
      --demo_type=${demo_type} --track # additional tag for logging purposes on wandb
  else
    echo "Demo path $demo_path does not exist. Skipping PushCube-v1 for ${demo_type}."
  fi
done