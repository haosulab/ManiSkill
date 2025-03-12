seeds=(3 6)
steps=(60 160)

for step in ${steps[@]}
do
  for seed in ${seeds[@]}
  do
    echo "📢 ppo_state with ${step} steps, seed ${seed}"
    python ppo_fast.py --env_id="PickCube-v1" --robot_uids="widowxai" --seed=${seed} \
      --num_envs=4096 --update_epochs=8 --num_minibatches=32 \
      --num_steps=${step} --num_eval_steps=${step} \
      --total_timesteps=100_000_000 \
      --num_eval_envs=16 \
      --compile --exp-name="ppo-state-pickcube-${seed}-step${step}" \
      --track
  done

  for seed in ${seeds[@]}
  do
    echo "📢 ppo_rgb with ${step} steps, seed ${seed}"
    python ppo_rgb.py --env_id="PickCube-v1" --robot_uids="widowxai_cam" --seed=${seed} \
      --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
      --num_steps=${step} --num_eval_steps=${step} \
      --total_timesteps=100_000_000 \
      --num_eval_envs=16 \
      --exp-name="ppo-rgb-pickcube-${seed}-step${step}" \
      --track
  done
done
