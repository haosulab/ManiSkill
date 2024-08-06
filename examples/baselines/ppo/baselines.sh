# for fair and correct RL evaluation against most RL algorithms, we do not do partial environment / early resets (which speed up training)
# moreover evaluation environments will reconfigure each reset in order to randomize the task completely as some
# tasks have different objects which are not changed at during normal resets.
# Furthermore, because of how these are evaluated, the hyperparameters here are tuned differently compared to with partial resets

seeds=(9351 4796 1788)

### State Based PPO Baselines ###
for seed in ${seeds[@]}
do
  python ppo.py --env_id="PushCube-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 --reward_scale=1 \
    --total_timesteps=50_000_000 \
    --no_partial_reset --reconfiguration_freq=1 --num_eval_envs=16 \
    --exp-name="ppo-PushCube-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo.py --env_id="PickCube-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 --reward_scale=1 \
    --total_timesteps=50_000_000 \
    --no_partial_reset --reconfiguration_freq=1 --num_eval_envs=16 \
    --exp-name="ppo-PickCube-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo.py --env_id="PickSingleYCB-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 --reward_scale=1 \
    --total_timesteps=50_000_000 \
    --no_partial_reset --reconfiguration_freq=1 --num_eval_envs=16 \
    --exp-name="ppo-PickSingleYCB-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo.py --env_id="PushT-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 --gamma=0.99 --reward_scale=0.1 \
    --total_timesteps=50_000_000 --num-steps=100 --num_eval_steps=100 \
    --no_partial_reset --reconfiguration_freq=1 --num_eval_envs=16 \
    --exp-name="ppo-PushT-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo.py --env_id="AnymalC-Reach-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 --gamma=0.99 --gae_lambda=0.95 --reward_scale=0.1 \
    --total_timesteps=50_000_000 --num-steps=200 --num-eval-steps=200 \
    --no_partial_reset --reconfiguration_freq=1 --num_eval_envs=16 \
    --exp-name="ppo-AnymalC-Reach-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

### RGB Based PPO Baselines ###
for seed in ${seeds[@]}
do
  python ppo_rgb.py --env_id="PushCube-v1" --seed=${seed} \
    --num_envs=256 --update_epochs=8 --num_minibatches=8 --reward_scale=1 \
    --total_timesteps=50_000_000 \
    --no_partial_reset --reconfiguration_freq=1 --num_eval_envs=16 \
    --exp-name="ppo-PushCube-v1-rgb-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo_rgb.py --env_id="PickCube-v1" --seed=${seed} \
    --num_envs=256 --update_epochs=8 --num_minibatches=8 --reward_scale=1 \
    --total_timesteps=50_000_000 \
    --no_partial_reset --reconfiguration_freq=1 --num_eval_envs=16 \
    --exp-name="ppo-PickCube-v1-rgb-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo_rgb.py --env_id="AnymalC-Reach-v1" --seed=${seed} \
    --num_envs=256 --update_epochs=8 --num_minibatches=32 --reward_scale=0.1 \
    --total_timesteps=50_000_000 --num-steps=200 --num-eval-steps=200 \
    --gamma=0.99 --gae_lambda=0.95 \
    --no_partial_reset --reconfiguration_freq=1 --num_eval_envs=16 \
    --exp-name="ppo-AnymalC-Reach-v1-rgb-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo_rgb.py --env_id="PushT-v1" --seed=${seed} \
    --num_envs=256 --update_epochs=8 --num_minibatches=8 --reward_scale=0.1 \
    --total_timesteps=50_000_000 --num-steps=100 --num_eval_steps=100 --gamma=0.99 \
    --no_partial_reset --reconfiguration_freq=1 --num_eval_envs=16 \
    --exp-name="ppo-PushT-v1-rgb-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
  done