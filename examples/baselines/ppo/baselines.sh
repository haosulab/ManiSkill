# Baseline results for PPO

seeds=(9351 4796 1788)

### State Based PPO Baselines ###
for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="PushCube-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-PushCube-v1-state-${seed}-walltime_efficient" \
    --track
done

# pick cube tests for ensuring manipulation robots work
for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="PickCube-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-PickCube-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="PickCubeSO100-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=8 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-PickCubeSO100-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="PickCubeWidowXAI-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=8 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-PickCubeWidowXAI-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="PushT-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 --gamma=0.99 \
    --total_timesteps=50_000_000 --num_eval_steps=100 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-PushT-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="StackCube-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-StackCube-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="RollBall-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 --gamma=0.95 \
    --total_timesteps=50_000_000 --num-eval-steps=80 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-RollBall-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="PullCube-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-PullCube-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="PokeCube-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-PokeCube-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="LiftPegUpright-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-LiftPegUpright-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="AnymalC-Reach-v1" --seed=${seed} \
    --num_envs=4096 --update_epochs=8 --num_minibatches=32 --gamma=0.99 --gae_lambda=0.95 \
    --total_timesteps=50_000_000 --num-steps=16 --num-eval-steps=200 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-AnymalC-Reach-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="PegInsertionSide-v1" --seed=${seed} \
    --num_envs=2048 --update_epochs=8 --num_minibatches=32 --gamma=0.97 --gae_lambda=0.95 \
    --total_timesteps=75_000_000 --num-steps=16 --num-eval-steps=100 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-PegInsertionSide-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="TwoRobotPickCube-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 --num-steps=100 --num-eval-steps=100 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-TwoRobotPickCube-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="UnitreeG1PlaceAppleInBowl-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 --num-steps=32 --num-eval-steps=100 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-UnitreeG1PlaceAppleInBowl-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="UnitreeG1TransportBox-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=100_000_000 --num-steps=32 --num-eval-steps=100 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-UnitreeG1TransportBox-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do 
  python ppo_fast.py --env_id="OpenCabinetDrawer-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 --num-steps=16 --num-eval-steps=100 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-OpenCabinetDrawer-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="PickSingleYCB-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-PickSingleYCB-v1-state-${seed}-walltime_efficient" \
    --track
done

### RGB Based PPO Baselines ###
for seed in ${seeds[@]}
do
  python ppo_rgb.py --env_id="PushCube-v1" --seed=${seed} \
    --num_envs=256 --update_epochs=8 --num_minibatches=8 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --exp-name="ppo-PushCube-v1-rgb-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ppo_rgb.py --env_id="PickCube-v1" --seed=${seed} \
    --num_envs=1024 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --exp-name="ppo-PickCube-v1-rgb-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ppo_rgb.py --env_id="PushT-v1" --seed=${seed} \
    --num_envs=1024 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 --num_eval_steps=100 --gamma=0.99 \
    --num_eval_envs=16 \
    --exp-name="ppo-PushT-v1-rgb-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do 
  python ppo_rgb.py --env_id="AnymalC-Reach-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 --gamma=0.99 --gae_lambda=0.95 \
    --total_timesteps=50_000_000 --num-steps=16 --num-eval-steps=200 \
    --num_eval_envs=16 --eval-reconfiguration-freq=0 \
    --exp-name="ppo-AnymalC-Reach-v1-rgb-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ppo_rgb.py --env_id="PickSingleYCB-v1" --seed=${seed} \
    --num_envs=1024 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --exp-name="ppo-PickSingleYCB-v1-rgb-${seed}-walltime_efficient" \
    --track
done