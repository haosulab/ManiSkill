# Baseline results for RFCL

env_id=PickCube-v1
demos=5 # number of demos to train on
seed=42
XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py configs/base_sac_ms3.yml \
  logger.exp_name=rfcl-${env_id}-state-${demos}_motionplanning_demos-${seed}-walltime_efficient logger.wandb=True \
  seed=${seed} train.num_demos=${demos} train.steps=1_000_000 \
  env.env_id=${env_id} \
  train.dataset_path="~/.maniskill/demos/${env_id}/motionplanning/trajectory.state.pd_joint_delta_pos.cpu.h5" \
  demo_type="motionplanning" config_type="walltime_efficient"\

env_id=PickCube-v1
demos=5 # number of demos to train on
seed=42
XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py configs/base_sac_ms3_sample_efficient.yml \
  logger.exp_name=rfcl-${env_id}-state-${demos}_motionplanning_demos-${seed}-sample_efficient logger.wandb=True \
  seed=${seed} train.num_demos=${demos} train.steps=1_000_000 \
  env.env_id=${env_id} \
  train.dataset_path="~/.maniskill/demos/${env_id}/motionplanning/trajectory.state.pd_joint_delta_pos.cpu.h5" \
  demo_type="motionplanning" config_type="sample_efficient"