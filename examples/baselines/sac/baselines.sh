#!/usr/bin/bash

seeds=(9351 4796 1788)

for seed in "${seeds[@]}"
do
    env_id=PickCube-v1
    num_envs=1024
    batch_size=4096
    horizon=50
    python sac.py --env_id="$env_id" --seed="$seed" \
        --total_timesteps=2_000_000 \
        --num_envs=$num_envs \
        --num_eval_envs=16 \
        --control_mode=pd_joint_delta_pos \
        --num_steps=$horizon \
        --num_eval_steps=$horizon \
        \
        --buffer_size=100_000 \
        --batch_size=$batch_size \
        --learning_starts=$((num_envs * 128)) \
        \
        --obs_rms \
        \
        --steps_per_env_per_iteration=1 \
        --grad_steps_per_iteration=10 \
        --policy_frequency=1 \
        --target_network_frequency=2 \
        \
        --policy_lr=5e-4 \
        --q_lr=5e-4 \
        --q_layer_norm \
        --min_q=2 \
        --num_q=2 \
        \
        --tau=0.05 \
        --alpha=1.0 \
        --autotune \
        --alpha_lr=5e-3 \
        \
        --gamma=0.8 \
        --bootstrap_at_done="always" \
        \
        --log_freq=$((num_envs * horizon)) \
        --eval_freq=$((num_envs * horizon)) \
        --exp-name="sac-${env_id}-state-${seed}-walltime_efficient" \
        --track

    env_id="PickSingleYCB-v1" 
    num_envs=1024
    batch_size=4096
    horizon=50
    python sac.py --env_id="$env_id" --seed="$seed" \
        --total_timesteps=50_000_000 \
        --num_envs=$num_envs \
        --num_eval_envs=16 \
        --control_mode=pd_joint_delta_pos \
        --num_steps=$horizon \
        --num_eval_steps=$horizon \
        \
        --buffer_size=1_000_000 \
        --batch_size=$batch_size \
        --learning_starts=$((num_envs * 32)) \
        \
        --steps_per_env_per_iteration=1 \
        --grad_steps_per_iteration=20 \
        --policy_frequency=1 \
        --target_network_frequency=2 \
        \
        --policy_lr=3e-4 \
        --q_lr=3e-4 \
        --q_layer_norm \
        --min_q=2 \
        --num_q=2 \
        \
        --tau=0.005 \
        --alpha=1.0 \
        --autotune \
        --alpha_lr=3e-4 \
        \
        --gamma=0.8 \
        --bootstrap_at_done="always" \
        \
        --log_freq=$((num_envs * horizon * 2)) \
        --eval_freq=$((num_envs * horizon * 4)) \
        --exp-name="sac-${env_id}-state-${seed}-walltime_efficient" \
        --track

    env_id="PegInsertionSide-v1" 
    num_envs=1024
    batch_size=4096
    horizon=100
    python sac.py --env_id="$env_id" --seed="$seed" \
        --total_timesteps=50_000_000 \
        --num_envs=$num_envs \
        --num_eval_envs=16 \
        --control_mode=pd_joint_delta_pos \
        --num_steps=$horizon \
        --num_eval_steps=$horizon \
        \
        --buffer_size=1_000_000 \
        --batch_size=$batch_size \
        --learning_starts=$((num_envs * 32)) \
        \
        --steps_per_env_per_iteration=1 \
        --grad_steps_per_iteration=20 \
        --policy_frequency=5 \
        --target_network_frequency=2 \
        \
        --policy_lr=3e-4 \
        --q_lr=3e-4 \
        --q_layer_norm \
        --min_q=2 \
        --num_q=2 \
        \
        --tau=0.005 \
        --alpha=1.0 \
        --autotune \
        --alpha_lr=3e-4 \
        \
        --gamma=0.99 \
        --bootstrap_at_done="always" \
        \
        --log_freq=$((num_envs * horizon * 2)) \
        --eval_freq=$((num_envs * horizon * 4)) \
        --exp-name="sac-${env_id}-state-${seed}-walltime_efficient" \
        --track

    env_id="UnitreeG1TransportBox-v1" 
    num_envs=4096
    batch_size=2048
    horizon=100
    python sac.py --env_id="$env_id" --seed="$seed" \
        --total_timesteps=50_000_000 \
        --num_envs=$num_envs \
        --num_eval_envs=16 \
        --control_mode=pd_joint_delta_pos \
        --num_steps=$horizon \
        --num_eval_steps=$horizon \
        \
        --buffer_size=1_000_000 \
        --batch_size=$batch_size \
        --learning_starts=$((num_envs * 32)) \
        \
        --steps_per_env_per_iteration=1 \
        --grad_steps_per_iteration=20 \
        --policy_frequency=1 \
        --target_network_frequency=2 \
        \
        --policy_lr=3e-4 \
        --q_lr=3e-4 \
        --q_layer_norm \
        --min_q=2 \
        --num_q=2 \
        \
        --tau=0.005 \
        --alpha=1.0 \
        --autotune \
        --alpha_lr=3e-4 \
        \
        --gamma=0.8 \
        --bootstrap_at_done="always" \
        \
        --log_freq=$((num_envs * horizon * 2)) \
        --eval_freq=$((num_envs * horizon * 4)) \
        --exp-name="sac-${env_id}-state-${seed}-walltime_efficient" \
        --track
done

