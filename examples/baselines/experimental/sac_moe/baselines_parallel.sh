#!/bin/bash

seeds=(9351 4796 1788)
file_name="sac_moe"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
MAX_PROCS_PER_GPU=1

MAX_PARALLEL=$((NUM_GPUS * MAX_PROCS_PER_GPU))

envs=(
  PushCube-v1 PickCube-v1 PickCubeSO100-v1 PushT-v1 StackCube-v1 RollBall-v1 PullCube-v1
  PokeCube-v1 LiftPegUpright-v1 AnymalC-Reach-v1 PegInsertionSide-v1 TwoRobotPickCube-v1
  UnitreeG1PlaceAppleInBowl-v1 UnitreeG1TransportBox-v1 OpenCabinetDrawer-v1 PickSingleYCB-v1
)

commands=()
for env in "${envs[@]}"; do
  for seed in "${seeds[@]}"; do
    cmd="python ${file_name}.py --env_id=\"$env\" --seed=$seed \
      --num_envs=32 --utd=0.5 --buffer_size=500_000 \
      --total_timesteps=1_000_000 --eval_freq=50_000 \
      --control-mode=\"pd_ee_delta_pos\" \
      --exp-name=\"${file_name}-${env}-state-${seed}-walltime_efficient\" --track"
    commands+=("$cmd")
  done
done

i=0
total=${#commands[@]}
pids=()

for cmd in "${commands[@]}"; do
  gpu_id=$(( (i / MAX_PROCS_PER_GPU) % NUM_GPUS ))

  echo "[GPU $gpu_id] $cmd"

  CUDA_VISIBLE_DEVICES=$gpu_id bash -c "$cmd" &

  pids+=($!)
  ((i++))

  if (( i % MAX_PARALLEL == 0 )); then
    wait "${pids[@]}"
    pids=()
  fi
done

wait "${pids[@]}"

rgbd_envs=("PushCube-v1" "PickCube-v1" "PushT-v1" "AnymalC-Reach-v1" "PickSingleYCB-v1")
rgbd_commands=()
for env in "${rgbd_envs[@]}"; do
  for seed in "${seeds[@]}"; do
    cmd="python ${file_name}_rgbd.py --env_id=\"$env\" --seed=$seed \
      --num_envs=32 --utd=0.5 --buffer_size=500_000 --obs_mode="rgb" --camera_width=64 --camera_height=64 \
      --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode=\"pd_ee_delta_pos\" \
      --exp-name=\"${file_name}-${env}-rgb-${seed}-walltime_efficient\" --track"
    rgbd_commands+=("$cmd")
  done
done

i=0
pids=()
echo "Starting RGBD configurations..."
for cmd in "${rgbd_commands[@]}"; do
  gpu_id=$(( (i / MAX_PROCS_PER_GPU) % NUM_GPUS ))
  echo "[GPU $gpu_id] $cmd"
  CUDA_VISIBLE_DEVICES=$gpu_id bash -c "$cmd" &
  pids+=($!)
  ((i++))
  
  if (( i % MAX_PARALLEL == 0 )); then
    wait "${pids[@]}"
    pids=()
  fi
done
wait "${pids[@]}"

echo "All jobs completed."