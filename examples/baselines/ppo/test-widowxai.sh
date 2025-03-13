if [ -z "$1" ]; then
  full_id=$(uuidgen)
  run_id=${full_id:0:6}
else
  run_id=$1
fi
echo "Using run ID: ${run_id}"
seeds=(9, 12, 420)
steps=(60 160)
control_modes=(
  "pd_joint_delta_pos"
  "pd_joint_pos"
  "pd_ee_delta_pos"
  "pd_ee_delta_pose"
  "pd_ee_pose"
  "pd_joint_target_delta_pos"
  "pd_ee_target_delta_pos"
  "pd_ee_target_delta_pose"
  "pd_joint_vel"
  "pd_joint_pos_vel"
  "pd_joint_delta_pos_vel"
)
for seed in ${seeds[@]}
do
  for step in ${steps[@]}
  do
    for control_mode in ${control_modes[@]}
    do
      echo "📢 ppo_state with ${step} steps, seed ${seed}, control mode ${control_mode}"
      python ppo_fast.py --env_id="PickCube-v1" --robot_uids="widowxai" --seed=${seed} \
      --num_envs=4096 --update_epochs=8 --num_minibatches=32 \
      --num_steps=${step} --num_eval_steps=${step} \
      --total_timesteps=80_000_000 \
      --num_eval_envs=16 \
      --control_mode=${control_mode} \
      --compile --exp-name="ppo-state-pickcube-${seed}-step${step}-${run_id}-${control_mode}" \
      --track
    #   echo "📢 ppo_rgb with ${step} steps, seed ${seed}, control mode ${control_mode}"
    #   python ppo_rgb.py --env_id="PickCube-v1" --robot_uids="widowxai_cam" --seed=${seed} \
    #   --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    #   --num_steps=${step} --num_eval_steps=${step} \
    #   --total_timesteps=60_000_000 \
    #   --num_eval_envs=16 \
    #   --control_mode=${control_mode} \
    #   --exp-name="ppo-rgb-pickcube-${seed}-step${step}-${run_id}-${control_mode}" \
    #   --track
    done
  done
done