seed=42
demos=100
for env_id in PickCube-v1
do
  python train.py --env-id ${env_id} --max_episode_steps 100 --total_iters 50000 \
    --control-mode "pd_ee_delta_pose" --num-demos ${demos} --seed ${seed} \
    --demo-path ~/.maniskill/demos/${env_id}/motionplanning/trajectory.state.pd_ee_delta_pose.h5 \
    --exp-name diffusion_policy-${env_id}-state-${demos}_motionplanning_demos-${seed} \
    --demo_type="motionplanning" --track # additional tag for logging purposes on wandb
done