
seeds=(9351 4796 1788)
for seed in ${seeds[@]}
do
    python ppo_orig.py --env_id="PickCube-v1" --seed=${seed} \
        --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
        --total_timesteps=10_000_000 \
        --num_eval_envs=16 \
        --exp-name="ppo-PickCube-v1-state-${seed}-walltime_efficient" \
        --wandb_entity="stonet2000" --track --wandb_project_name="PPO-ManiSkill-GPU-SpeedRun"

    python ppo_orig.py --env_id="PickCube-v1" --seed=${seed} \
        --num_envs=4096 --update_epochs=12 --num_minibatches=32 \
        --total_timesteps=10_000_000 \
        --num_eval_envs=16 \
        --num-steps=10 --eval-freq=10 --no-finite-horizon-gae \
        --exp-name="ppo-PickCube-v1-state-${seed}-no-finite-horizon-gae-4096envs-10steps-walltime_efficient" \
        --wandb_entity="stonet2000" --track --wandb_project_name="PPO-ManiSkill-GPU-SpeedRun"

    python ppo_leanrl.py --env_id="PickCube-v1" --seed=${seed} \
        --num_envs=4096 --update_epochs=12 --num_minibatches=32 \
        --total_timesteps=10_000_000 \
        --num_eval_envs=16 \
        --num-steps=10 --eval-freq=10 \
        --exp-name="ppo-leanrl-compile-PickCube-v1-state-${seed}-no-finite-horizon-gae-4096envs-10steps-walltime_efficient" \
        --wandb_entity="stonet2000" --track --wandb_project_name="PPO-ManiSkill-GPU-SpeedRun" --compile

    python ppo_leanrl.py --env_id="PickCube-v1" --seed=${seed} \
        --num_envs=4096 --update_epochs=12 --num_minibatches=32 \
        --total_timesteps=10_000_000 \
        --num_eval_envs=16 \
        --num-steps=10 --eval-freq=10 \
        --exp-name="ppo-leanrl-compile-cudagraphs-PickCube-v1-state-${seed}-no-finite-horizon-gae-4096envs-10steps-walltime_efficient" \
        --wandb_entity="stonet2000" --wandb_project_name="PPO-ManiSkill-GPU-SpeedRun" --compile
done


# seeds=(9351 4796 1788)
for seed in ${seeds[@]}
do
    

    python ppo_orig.py --env_id="PushT-v1" --seed=${seed} \
        --num_envs=4096 --update_epochs=8 --num_minibatches=32 --gamma=0.99 \
        --total_timesteps=50_000_000 \
        --num_eval_envs=16 \
        --no-finite-horizon-gae --num-steps=25 --num_eval_steps=100 --gamma=0.99 --eval-freq=10 \
        --exp-name="ppo-PushT-v1-state-${seed}-no-finite-horizon-gae-4096envs-25steps-walltime_efficient" \
        --wandb_entity="stonet2000" --track --wandb_project_name="PPO-ManiSkill-GPU-SpeedRun"
    
    python ppo_orig.py --env_id="PushT-v1" --seed=${seed} \
        --num_envs=1024 --update_epochs=8 --num_minibatches=32 --gamma=0.99 \
        --total_timesteps=50_000_000 --num-steps=100 --num_eval_steps=100 \
        --num_eval_envs=16 \
        --exp-name="ppo-PushT-v1-state-${seed}-walltime_efficient" \
        --wandb_entity="stonet2000" --track --wandb_project_name="PPO-ManiSkill-GPU-SpeedRun"
done
