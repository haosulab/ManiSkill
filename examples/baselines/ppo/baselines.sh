# Baseline results for PPO

seeds=(9351 4796 1788)

### State Based PPO Baselines ###
for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="PushCube-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="ppo-PushCube-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="PickCube-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="ppo-PickCube-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="PushT-v1" --seed=${seed} \
    --num_envs=4096 --update_epochs=8 --num_minibatches=32 --gamma=0.99 \
    --total_timesteps=50_000_000 --num-steps=16 --num_eval_steps=100 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="ppo-PushT-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="StackCube-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="ppo-StackCube-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="RollBall-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 --gamma=0.95 \
    --total_timesteps=50_000_000 --num-eval-steps=80 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="ppo-RollBall-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="PullCube-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="ppo-PullCube-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="PokeCube-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="ppo-PokeCube-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="LiftPegUpright-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="ppo-LiftPegUpright-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="AnymalC-Reach-v1" --seed=${seed} \
    --num_envs=4096 --update_epochs=8 --num_minibatches=32 --gamma=0.99 --gae_lambda=0.95 \
    --total_timesteps=50_000_000 --num-steps=16 --num-eval-steps=200 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="ppo-AnymalC-Reach-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="PegInsertionSide-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 --gamma=0.97 --gae_lambda=0.95 \
    --total_timesteps=150_000_000 --num-steps=100 --num-eval-steps=100 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="ppo-PegInsertionSide-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="TwoRobotPickCube-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 --num-steps=100 --num-eval-steps=100 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="ppo-TwoRobotPickCube-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="UnitreeG1PlaceAppleInBowl-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 --num-steps=32 --num-eval-steps=100 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="ppo-UnitreeG1PlaceAppleInBowl-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="UnitreeG1TransportBox-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=100_000_000 --num-steps=32 --num-eval-steps=100 \
    --num_eval_envs=16 \
    --save-model --cudagraphs --exp-name="ppo-UnitreeG1TransportBox-v1-state-${seed}-walltime_efficient" \
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