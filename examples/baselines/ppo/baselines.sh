# Baseline results for PPO

seeds=(9351 4796 1788)

### State Based PPO Baselines ###


for seed in ${seeds[@]}
do
  python ppo.py --env_id="PickCube-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --exp-name="ppo-PickCube-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo.py --env_id="PickSingleYCB-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --exp-name="ppo-PickSingleYCB-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo.py --env_id="PushT-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 --gamma=0.99 \
    --total_timesteps=50_000_000 --num-steps=100 --num_eval_steps=100 \
    --num_eval_envs=16 \
    --exp-name="ppo-PushT-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  # no partial reset is used to help get higher success_at_end success rates
  python ppo.py --env_id="AnymalC-Reach-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 --gamma=0.99 --gae_lambda=0.95 \
    --total_timesteps=50_000_000 --num-steps=200 --num-eval-steps=200 \
    --num_eval_envs=16 --no-partial-reset \
    --exp-name="ppo-AnymalC-Reach-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo.py --env_id="PushCube-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --exp-name="ppo-PushCube-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

### RGB Based PPO Baselines ###
for seed in ${seeds[@]}
do
  python ppo_rgb.py --env_id="PushCube-v1" --seed=${seed} \
    --num_envs=256 --update_epochs=8 --num_minibatches=8 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --exp-name="ppo-PushCube-v1-rgb-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo_rgb.py --env_id="PickCube-v1" --seed=${seed} \
    --num_envs=256 --update_epochs=8 --num_minibatches=8 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --exp-name="ppo-PickCube-v1-rgb-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  # no partial reset is used to help get higher success_at_end success rates
  python ppo_rgb.py --env_id="AnymalC-Reach-v1" --seed=${seed} \
    --num_envs=256 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 --num-steps=200 --num-eval-steps=200 \
    --gamma=0.99 --gae_lambda=0.95 \
    --num_eval_envs=16 --no-partial-reset \
    --exp-name="ppo-AnymalC-Reach-v1-rgb-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo_rgb.py --env_id="PushT-v1" --seed=${seed} \
    --num_envs=256 --update_epochs=8 --num_minibatches=8 \
    --total_timesteps=50_000_000 --num-steps=100 --num_eval_steps=100 --gamma=0.99 \
    --num_eval_envs=16 \
    --exp-name="ppo-PushT-v1-rgb-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
  done