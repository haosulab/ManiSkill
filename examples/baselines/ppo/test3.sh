# seeds=(2371 3462 4553 5644 6735 7826 8917)
seeds=(9351 4796 1788 2371 3462 4553 5644 6735 7826 8917)
for seed in ${seeds[@]}
do
    for num_envs in 4096
    do
        for update_epochs in 12
        do
            for num_steps in 4
            do
                python ppo_leanrl.py --env_id="PickCube-v1" --seed=${seed} \
                    --num_envs=${num_envs} --update_epochs=${update_epochs} --num_minibatches=32 \
                    --total_timesteps=4_000_000 \
                    --num_eval_envs=16 \
                    --num-steps=${num_steps} --eval-freq=10 \
                    --exp-name="ppo-leanrl-PickCube-v1-state-${seed}-no-finite-horizon-gae-${num_envs}envs-${num_steps}-steps-${update_epochs}-update_epochs-walltime_efficient" \
                    --wandb_entity="stonet2000" --track --wandb_project_name="PPO-ManiSkill-GPU-SpeedRun" --no-capture_video
            done
        done
    done
done