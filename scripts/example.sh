python map_rl/train_ppo.py \
    --env_id=PickCubeDiscreteInit-v1 \
    --robot_uids=xarm6_robotiq \
    --control_mode=pd_joint_vel \
    --exp_name=PickCube_xarm6_ppo \
    --num_envs=50 \
    --num_eval_envs=20 \
    --eval_freq=20 \
    --total_timesteps=100_000_000 \
    --num_steps=100 \
    --gamma=0.9 \
    --capture-video \
    --track \
    --use_map \
    --use_local_fusion \
    --vision_encoder=plain_cnn \
    --wandb_project_name "PPO-RL-Map"


#plain-cnn / no map (wandb tag "plain-cnn-no-map")
#plain-cnn / map / no local fusion (wandb tag "plain-cnn-map-no-local-fusion")
#plain-cnn / map / local fusion (wandb tag "plain-cnn-map-local-fusion")
#dino / no map (wandb tag "dino-no-map")
#dino / map / no local fusion (wandb tag "dino-map-no-local-fusion")
#dino / map / local fusion (wandb tag "dino-map-local-fusion")