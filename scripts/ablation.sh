#!/usr/bin/env bash
set -euo pipefail

# Run all six configurations once, sequentially, with distinct wandb tags
# Project: PPO-RL-Map

cd "$(dirname "$0")/.."

COMMON_ARGS=(
  --env_id=PickCubeDiscreteInit-v1
  --robot_uids=xarm6_robotiq
  --control_mode=pd_joint_vel
  --num_envs=50
  --num_eval_envs=20
  --eval_freq=20
  --total_timesteps=100_000_000
  --num_steps=100
  --gamma=0.9
  --capture-video
  --track
  --wandb_project_name "PPO-RL-Map"
)

run_cfg() {
  local TAG="$1"; shift
  echo "=== Running: ${TAG} ==="
  python map_rl/train_ppo.py \
    "${COMMON_ARGS[@]}" \
    --exp_name=PickCube_xarm6_ppo__${TAG} \
    --wandb_tags ${TAG} \
    "$@"
}

# 1) plain-cnn / no map
run_cfg plain-cnn-no-map \
  --vision_encoder=plain_cnn

# 2) plain-cnn / map / no local fusion
run_cfg plain-cnn-map-no-local-fusion \
  --use_map \
  --vision_encoder=plain_cnn

# 3) plain-cnn / map / local fusion
run_cfg plain-cnn-map-local-fusion \
  --use_map \
  --use_local_fusion \
  --vision_encoder=plain_cnn

# 4) dino / no map
run_cfg dino-no-map \
  --vision_encoder=dino

# 5) dino / map / no local fusion
run_cfg dino-map-no-local-fusion \
  --use_map \
  --vision_encoder=dino

# 6) dino / map / local fusion
run_cfg dino-map-local-fusion \
  --use_map \
  --use_local_fusion \
  --vision_encoder=dino