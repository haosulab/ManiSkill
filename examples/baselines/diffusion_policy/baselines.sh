for env_id in PickCube-v1
do
  python train.py --env-id ${env_id} --max_episode_steps 100 \
    --control-mode "pd_joint_delta_pos" \
    --demo-path ~/.maniskill/demos/${env_id}/motionplanning/trajectory.state.pd_joint_delta_pos.h5 \
    --track
done