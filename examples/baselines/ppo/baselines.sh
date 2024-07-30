seeds=(9351 4796 1788)
for seed in ${seeds[@]}
do
  python ppo.py --env_id="PushCube-v1" --seed=${seed} \
    --num_envs=2048 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 --eval_freq=10 --num-steps=20 \
    --no_partial_reset \
    --exp-name="ppo-PushCube-v1-state-${seed}-walltime_efficient" \
    --wandb_entity="stonet2000" --track
done

for seed in ${seeds[@]}
do
  python ppo.py --env_id="PickCube-v1" --seed=${seed} \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --no_partial_reset \
    --exp-name="ppo-PickCube-v1-state-${seed}-walltime_efficient" \
    # --wandb_entity="stonet2000" --track
done