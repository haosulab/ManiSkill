#!/usr/bin/bash

seeds=(9351 4796 1788)


for seed in "${seeds[@]}"
do
  python ppo_fast.py --env_id="PickCube-v1" --seed="$seed" \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=8_000_000 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-PickCube-v1-state-${seed}-walltime_efficient" \
    --track

  python ppo_fast.py --env_id="PickSingleYCB-v1" --seed="$seed" \
    --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=23_000_000 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-PickSingleYCB-v1-state-${seed}-walltime_efficient" \
    --track

  python ppo_fast.py --env_id="PegInsertionSide-v1" --seed="$seed" \
    --num_envs=2048 --update_epochs=8 --num_minibatches=32 --gamma=0.97 --gae_lambda=0.95 \
    --total_timesteps=45_000_000 --num-steps=16 --num-eval-steps=100 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-PegInsertionSide-v1-state-${seed}-walltime_efficient" \
    --track

  python ppo_fast.py --env_id="UnitreeG1TransportBox-v1" --seed="$seed" \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=45_000_000 --num-steps=32 --num-eval-steps=100 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-UnitreeG1TransportBox-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in "${seeds[@]}"
do
  python ppo_fast.py --env_id="PickCube-v1" --seed="$seed" \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=8_000_000 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-PickCube-v1-state-${seed}-walltime_efficient-staggered_reset" \
    --track \
    --staggered_reset

  python ppo_fast.py --env_id="PickSingleYCB-v1" --seed="$seed" \
    --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=23_000_000 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-PickSingleYCB-v1-state-${seed}-walltime_efficient-staggered_reset" \
    --track \
    --staggered_reset

  python ppo_fast.py --env_id="PegInsertionSide-v1" --seed="$seed" \
    --num_envs=2048 --update_epochs=8 --num_minibatches=32 --gamma=0.97 --gae_lambda=0.95 \
    --total_timesteps=45_000_000 --num-steps=16 --num-eval-steps=100 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-PegInsertionSide-v1-state-${seed}-walltime_efficient-staggered_reset" \
    --track \
    --staggered_reset

  python ppo_fast.py --env_id="UnitreeG1TransportBox-v1" --seed="$seed" \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=45_000_000 --num-steps=32 --num-eval-steps=100 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-UnitreeG1TransportBox-v1-state-${seed}-walltime_efficient-staggered_reset" \
    --track \
    --staggered_reset
done
