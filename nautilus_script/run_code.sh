python mapping/generate_pose.py
python mapping/generate_mapping_data.py --grid-dim=15
python mapping/map_multi_table.py --save

python map_rl/ppo_rgb.py \
    --env_id=PickCubeDiscreteInit-v1 \
    --robot_uids=xarm6_robotiq \
    --control_mode=pd_joint_vel \
    --exp_name=PickCube_xarm6_ppo \
    --num_envs=32 \
    --num_eval_envs=20 \
    --eval_freq=20 \
    --total_timesteps=100_000_000 \
    --num_steps=50 \
    --gamma=0.8 \
    --capture-video \
    --track \
    --wandb_project_name "ManiSkill-RL" \
    --grid_dim=15

    