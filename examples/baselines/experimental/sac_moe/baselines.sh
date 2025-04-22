# Baseline results for beta playground

seeds=(9351 4796 1788)
file_name='sac_moe'

### State Based Beta Playground Baselines ###
for seed in ${seeds[@]}
do
  python ${file_name}.py --env_id="PushCube-v1" \
    --num_envs=32 --utd=0.5 --buffer_size=500_000  \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-PushCube-v1-state-${seed}-walltime_efficient" \
    --track
done

# pick cube tests for ensuring manipulation robots work
for seed in ${seeds[@]}
do
  python ${file_name}.py --env_id="PickCube-v1" --seed=${seed} \
    --num_envs=32 --utd=0.5 --buffer_size=500_000  \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-PickCube-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ${file_name}.py --env_id="PickCubeSO100-v1" --seed=${seed} \
    --num_envs=32 --utd=0.5 --buffer_size=500_000  \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-PickCubeSO100-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ${file_name}.py --env_id="PushT-v1" --seed=${seed} \
    --num_envs=32 --utd=0.5 --buffer_size=500_000  \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-PushT-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ${file_name}.py --env_id="StackCube-v1" --seed=${seed} \
    --num_envs=32 --utd=0.5 --buffer_size=500_000  \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-StackCube-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ${file_name}.py --env_id="RollBall-v1" --seed=${seed} \
    --num_envs=32 --utd=0.5 --buffer_size=500_000  \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-RollBall-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ${file_name}.py --env_id="PullCube-v1" --seed=${seed} \
    --num_envs=32 --utd=0.5 --buffer_size=500_000  \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-PullCube-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ${file_name}.py --env_id="PokeCube-v1" --seed=${seed} \
    --num_envs=32 --utd=0.5 --buffer_size=500_000  \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-PokeCube-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ${file_name}.py --env_id="LiftPegUpright-v1" --seed=${seed} \
    --num_envs=32 --utd=0.5 --buffer_size=500_000  \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-LiftPegUpright-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ${file_name}.py --env_id="AnymalC-Reach-v1" --seed=${seed} \
    --num_envs=32 --utd=0.5 --buffer_size=500_000  \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-AnymalC-Reach-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ${file_name}.py --env_id="PegInsertionSide-v1" --seed=${seed} \
    --num_envs=32 --utd=0.5 --buffer_size=500_000  \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-PegInsertionSide-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ${file_name}.py --env_id="TwoRobotPickCube-v1" --seed=${seed} \
    --num_envs=32 --utd=0.5 --buffer_size=500_000  \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-TwoRobotPickCube-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ${file_name}.py --env_id="UnitreeG1PlaceAppleInBowl-v1" --seed=${seed} \
    --num_envs=32 --utd=0.5 --buffer_size=500_000  \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-UnitreeG1PlaceAppleInBowl-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ${file_name}.py --env_id="UnitreeG1TransportBox-v1" --seed=${seed} \
    --num_envs=32 --utd=0.5 --buffer_size=500_000  \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-UnitreeG1TransportBox-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do 
  python ${file_name}.py --env_id="OpenCabinetDrawer-v1" --seed=${seed} \
    --num_envs=32 --utd=0.5 --buffer_size=500_000  \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-OpenCabinetDrawer-v1-state-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ${file_name}.py --env_id="PickSingleYCB-v1" --seed=${seed} \
    --num_envs=32 --utd=0.5 --buffer_size=500_000  \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-PickSingleYCB-v1-state-${seed}-walltime_efficient" \
    --track
done

exit

### RGB Based Beta Playground Baselines ###
for seed in ${seeds[@]}
do
  python ${file_name}_rgbd.py --env_id="PushCube-v1" --seed=${seed} \
    --num_envs=32 --utd=0.5 --buffer_size=500_000 --obs_mode="rgb" --camera_width=64 --camera_height=64 \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-PushCube-v1-rgb-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ${file_name}_rgbd.py --env_id="PickCube-v1" --seed=${seed} \
    --num_envs=32 --utd=0.5 --buffer_size=500_000 --obs_mode="rgb" --camera_width=64 --camera_height=64 \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-PickCube-v1-rgb-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ${file_name}_rgbd.py --env_id="PushT-v1" --seed=${seed} \
    --num_envs=32 --utd=0.5 --buffer_size=500_000 --obs_mode="rgb" --camera_width=64 --camera_height=64 \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-PushT-v1-rgb-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do 
  python ${file_name}_rgbd.py --env_id="AnymalC-Reach-v1" --seed=${seed} \
    --num_envs=32 --utd=0.5 --buffer_size=500_000 --obs_mode="rgb" --camera_width=64 --camera_height=64 \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-AnymalC-Reach-v1-rgb-${seed}-walltime_efficient" \
    --track
done

for seed in ${seeds[@]}
do
  python ${file_name}_rgbd.py --env_id="PickSingleYCB-v1" --seed=${seed} \
    --num_envs=32 --utd=0.5 --buffer_size=500_000 --obs_mode="rgb" --camera_width=64 --camera_height=64 \
    --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
    --exp-name="${file_name}-PickSingleYCB-v1-rgb-${seed}-walltime_efficient" \
    --track
done