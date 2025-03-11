seeds=(9351 4796 1788)


for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="PickCube-v1" --robot_uids="widowxai" --seed=${seed} \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="ppo-PickCube-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ppo_rgb.py --env_id="PickCube-v1" --robot_uids="widowxai_cam" --seed=${seed} \
    --num_envs=1024 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --exp-name="ppo-PickCube-v1-rgb-${seed}-walltime_efficient" \
    --track
done
