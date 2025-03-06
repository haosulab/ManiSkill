# Baseline results for TD-MPC2 (We recommend running individual experiments, instead of the entire file)

seed=(9351 4796 1788)
# Wandb settings 
use_wandb=false
wandb_entity="na"
wandb_project="na"
wandb_group="na"

### State Based TD-MPC2 Baselines ###

## walltime_efficient Setting ##

# PushCube-v1 #
for seed in ${seed[@]}
do 
    python train.py model_size=5 steps=1_000_000 seed=$seed buffer_size=1_000_000 exp_name=tdmpc2 \
        env_id=PushCube-v1 num_envs=32 control_mode=pd_joint_delta_pos env_type=gpu obs=state \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-PushCube-v1-state-$seed-walltime_efficient
done

# PickCube-v1 #
for seed in ${seed[@]}
do 
    python train.py model_size=5 steps=1_000_000 seed=$seed buffer_size=1_000_000 exp_name=tdmpc2 \
        env_id=PickCube-v1 num_envs=32 control_mode=pd_ee_delta_pos env_type=gpu obs=state \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-PickCube-v1-state-$seed-walltime_efficient
done

# StackCube-v1 #
for seed in ${seed[@]}
do 
    python train.py model_size=5 steps=2_000_000 seed=$seed buffer_size=1_000_000 exp_name=tdmpc2 \
        env_id=StackCube-v1 num_envs=32 control_mode=pd_joint_delta_pos env_type=gpu obs=state \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-StackCube-v1-state-$seed-walltime_efficient
done

# PegInsertionSide-v1 #
for seed in ${seed[@]}
do 
    python train.py model_size=5 steps=2_000_000 seed=$seed buffer_size=1_000_000 exp_name=tdmpc2 \
        env_id=PegInsertionSide-v1 num_envs=32 control_mode=pd_joint_delta_pos env_type=gpu obs=state \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-PegInsertionSide-v1-state-$seed-walltime_efficient
done

# PushT-v1 #
for seed in ${seed[@]}
do 
    python train.py model_size=5 steps=1_000_000 seed=$seed buffer_size=1_000_000 exp_name=tdmpc2 \
        env_id=PushT-v1 num_envs=32 control_mode=pd_joint_delta_pos env_type=gpu obs=state \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-PushT-v1-state-$seed-walltime_efficient
done

# AnymalC-Reach-v1 #

for seed in ${seed[@]}
do 
    echo y | python train.py model_size=5 steps=1_000_000 seed=$seed buffer_size=1_000_000 exp_name=tdmpc2 \
        env_id=AnymalC-Reach-v1 num_envs=32 control_mode=pd_joint_delta_pos env_type=gpu obs=state eval_reconfiguration_frequency=0 \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-AnymalC-Reach-v1-state-$seed-walltime_efficient
done

# UnitreeG1TransportBox-v1 #
for seed in ${seed[@]}
do 
    python train.py model_size=5 steps=1_000_000 seed=$seed buffer_size=1_000_000 exp_name=tdmpc2 \
        env_id=UnitreeG1TransportBox-v1 num_envs=32 control_mode=pd_joint_delta_pos env_type=gpu obs=state \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-UnitreeG1TransportBox-v1-state-$seed-walltime_efficient
done

## sample_efficient Setting ##

# PushCube-v1 #
for seed in ${seed[@]}
do 
    python train.py model_size=5 steps=1_000_000 seed=$seed buffer_size=1_000_000 exp_name=tdmpc2 \
        env_id=PushCube-v1 num_envs=1 control_mode=pd_ee_delta_pose env_type=cpu obs=state \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=sample_efficient \
        wandb_name=tdmpc2-PushCube-v1-state-$seed-sample_efficient
done

# PickCube-v1 #
for seed in ${seed[@]}
do 
    python train.py model_size=5 steps=1_000_000 seed=$seed buffer_size=1_000_000 exp_name=tdmpc2 \
        env_id=PickCube-v1 num_envs=1 control_mode=pd_ee_delta_pos env_type=cpu obs=state \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=sample_efficient \
        wandb_name=tdmpc2-PickCube-v1-state-$seed-sample_efficient
done

# StackCube-v1 #
for seed in ${seed[@]}
do 
    python train.py model_size=5 steps=2_000_000 seed=$seed buffer_size=1_000_000 exp_name=tdmpc2 \
        env_id=StackCube-v1 num_envs=1 control_mode=pd_ee_delta_pose env_type=cpu obs=state \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=sample_efficient \
        wandb_name=tdmpc2-StackCube-v1-state-$seed-sample_efficient
done

# PegInsertionSide-v1 #
for seed in ${seed[@]}
do 
    python train.py model_size=5 steps=2_000_000 seed=$seed buffer_size=1_000_000 exp_name=tdmpc2 \
        env_id=PegInsertionSide-v1 num_envs=1 control_mode=pd_ee_delta_pose env_type=cpu obs=state \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=sample_efficient \
        wandb_name=tdmpc2-PegInsertionSide-v1-state-$seed-sample_efficient
done

# PushT-v1 #
for seed in ${seed[@]}
do 
    python train.py model_size=5 steps=1_000_000 seed=$seed buffer_size=1_000_000 exp_name=tdmpc2 \
        env_id=PushT-v1 num_envs=1 control_mode=pd_ee_delta_pose env_type=cpu obs=state \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=sample_efficient \
        wandb_name=tdmpc2-PushT-v1-state-$seed-sample_efficient
done

# AnymalC-Reach-v1 #

for seed in ${seed[@]}
do 
    echo y | python train.py model_size=5 steps=1_000_000 seed=$seed buffer_size=1_000_000 exp_name=tdmpc2 \
        env_id=AnymalC-Reach-v1 num_envs=1 control_mode=pd_joint_delta_pos env_type=cpu obs=state eval_reconfiguration_frequency=0 \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=sample_efficient \
        wandb_name=tdmpc2-AnymalC-Reach-v1-state-$seed-sample_efficient
done

# UnitreeG1TransportBox-v1 #
for seed in ${seed[@]}
do 
    python train.py model_size=5 steps=1_000_000 seed=$seed buffer_size=1_000_000 exp_name=tdmpc2 \
        env_id=UnitreeG1TransportBox-v1 num_envs=1 control_mode=pd_joint_delta_pos env_type=cpu obs=state \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=sample_efficient \
        wandb_name=tdmpc2-UnitreeG1TransportBox-v1-state-$seed-sample_efficient
done


### RGB Based TD-MPC2 Baselines ###

## walltime_efficient Setting ##

# PushCube-v1 #
for seed in ${seed[@]}
do
    python train.py model_size=5 steps=1_000_000 seed=$seed buffer_size=100_000 exp_name=tdmpc2 \
        env_id=PushCube-v1 num_envs=32 control_mode=pd_joint_delta_pos env_type=gpu obs=rgb render_size=128 \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-PushCube-v1-rgb-$seed-walltime_efficient
done

# PickCube-v1 #
for seed in ${seed[@]}
do
    python train.py model_size=5 steps=1_000_000 seed=$seed buffer_size=100_000 exp_name=tdmpc2 \
        env_id=PickCube-v1 num_envs=32 control_mode=pd_ee_delta_pos env_type=gpu obs=rgb render_size=128 \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-PickCube-v1-rgb-$seed-walltime_efficient
done

# StackCube-v1 #
for seed in ${seed[@]}
do
    python train.py model_size=5 steps=4_000_000 seed=$seed buffer_size=100_000 exp_name=tdmpc2 \
        env_id=StackCube-v1 num_envs=32 control_mode=pd_joint_delta_pos env_type=gpu obs=rgb render_size=128 \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-StackCube-v1-rgb-$seed-walltime_efficient
done

# PegInsertionSide-v1 #
for seed in ${seed[@]}
do
    python train.py model_size=5 steps=4_000_000 seed=$seed buffer_size=100_000 exp_name=tdmpc2 \
        env_id=PegInsertionSide-v1 num_envs=32 control_mode=pd_joint_delta_pos env_type=gpu obs=rgb render_size=128 \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-PegInsertionSide-v1-rgb-$seed-walltime_efficient
done

# PushT-v1 #
for seed in ${seed[@]}
do
    python train.py model_size=5 steps=2_000_000 seed=$seed buffer_size=100_000 exp_name=tdmpc2 \
        env_id=PushT-v1 num_envs=32 control_mode=pd_joint_delta_pos env_type=gpu obs=rgb render_size=128 \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-PushT-v1-rgb-$seed-walltime_efficient
done

# AnymalC-Reach-v1 #
for seed in ${seed[@]}
do
    echo y | python train.py model_size=5 steps=2_000_000 seed=$seed buffer_size=100_000 exp_name=tdmpc2 \
        env_id=AnymalC-Reach-v1 num_envs=32 control_mode=pd_joint_delta_pos env_type=gpu obs=rgb render_size=128 eval_reconfiguration_frequency=0 \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-AnymalC-Reach-v1-rgb-$seed-walltime_efficient
done

# UnitreeG1TransportBox-v1 #
for seed in ${seed[@]}
do
    python train.py model_size=5 steps=2_000_000 seed=$seed buffer_size=100_000 exp_name=tdmpc2 \
        env_id=UnitreeG1TransportBox-v1 num_envs=32 control_mode=pd_joint_delta_pos env_type=gpu obs=rgb render_size=128 \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-UnitreeG1TransportBox-v1-rgb-$seed-walltime_efficient
done

## sample_efficient Setting ##

# PushCube-v1 #
for seed in ${seed[@]}
do
    python train.py model_size=5 steps=1_000_000 seed=$seed buffer_size=100_000 exp_name=tdmpc2 \
        env_id=PushCube-v1 num_envs=1 control_mode=pd_ee_delta_pose env_type=cpu obs=rgb render_size=128 \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-PushCube-v1-rgb-$seed-walltime_efficient
done

# PickCube-v1 #
for seed in ${seed[@]}
do
    python train.py model_size=5 steps=1_000_000 seed=$seed buffer_size=100_000 exp_name=tdmpc2 \
        env_id=PickCube-v1 num_envs=1 control_mode=pd_ee_delta_pos env_type=cpu obs=rgb render_size=128 \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-PickCube-v1-rgb-$seed-walltime_efficient
done

# StackCube-v1 #
for seed in ${seed[@]}
do
    python train.py model_size=5 steps=4_000_000 seed=$seed buffer_size=100_000 exp_name=tdmpc2 \
        env_id=StackCube-v1 num_envs=1 control_mode=pd_ee_delta_pose env_type=cpu obs=rgb render_size=128 \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-StackCube-v1-rgb-$seed-walltime_efficient
done

# PegInsertionSide-v1 #
for seed in ${seed[@]}
do
    python train.py model_size=5 steps=4_000_000 seed=$seed buffer_size=100_000 exp_name=tdmpc2 \
        env_id=PegInsertionSide-v1 num_envs=1 control_mode=pd_ee_delta_pose env_type=cpu obs=rgb render_size=128 \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-PegInsertionSide-v1-rgb-$seed-walltime_efficient
done

# PushT-v1 #
for seed in ${seed[@]}
do
    python train.py model_size=5 steps=2_000_000 seed=$seed buffer_size=100_000 exp_name=tdmpc2 \
        env_id=PushT-v1 num_envs=1 control_mode=pd_ee_delta_pose env_type=cpu obs=rgb render_size=128 \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-PushT-v1-rgb-$seed-walltime_efficient
done

# AnymalC-Reach-v1 #
for seed in ${seed[@]}
do
    echo y | python train.py model_size=5 steps=2_000_000 seed=$seed buffer_size=100_000 exp_name=tdmpc2 \
        env_id=AnymalC-Reach-v1 num_envs=1 control_mode=pd_joint_delta_pos env_type=cpu obs=rgb render_size=128 eval_reconfiguration_frequency=0 \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-AnymalC-Reach-v1-rgb-$seed-walltime_efficient
done

# UnitreeG1TransportBox-v1 #
for seed in ${seed[@]}
do
    python train.py model_size=5 steps=2_000_000 seed=$seed buffer_size=100_000 exp_name=tdmpc2 \
        env_id=UnitreeG1TransportBox-v1 num_envs=1 control_mode=pd_joint_delta_pos env_type=cpu obs=rgb render_size=128 \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
        wandb_name=tdmpc2-UnitreeG1TransportBox-v1-rgb-$seed-walltime_efficient
done