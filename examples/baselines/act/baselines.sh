seed=1
# State based
for demos in 100; do
  python train.py --env-id PickCube-v1 \
    --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
    --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num_demos $demos --max_episode_steps 100 \
    --total_iters 30000 --log_freq 100 --eval_freq 5000 \
    --exp-name=act-PickCube-v1-state-${demos}_motionplanning_demos-$seed \
    --demo_type motionplanning --track

  python train.py --env-id PushCube-v1 \
    --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
    --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num_demos $demos --max_episode_steps 100 \
    --total_iters 30000 --log_freq 100 --eval_freq 5000 \
    --exp-name=act-PushCube-v1-state-${demos}_motionplanning_demos-$seed \
    --demo_type motionplanning --track

  python train.py --env-id StackCube-v1 \
    --demo-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
    --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num_demos $demos --max_episode_steps 200 \
    --total_iters 30000 --log_freq 100 --eval_freq 5000 \
    --exp-name=act-StackCube-v1-state-${demos}_motionplanning_demos-$seed \
    --demo_type motionplanning --track

  python train.py --env-id PegInsertionSide-v1 \
    --demo-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.h5 \
    --control-mode "pd_ee_delta_pose" --sim-backend "physx_cpu" --num_demos $demos --max_episode_steps 300 \
    --total_iters 100000 --log_freq 100 --eval_freq 5000 \
    --exp-name=act-PegInsertionSide-v1-state-${demos}_motionplanning_demos-$seed \
    --demo_type motionplanning --track

  python train.py --env-id PushT-v1 \
    --demo-path ~/.maniskill/demos/PushT-v1/rl/trajectory.state.pd_ee_delta_pose.physx_cuda.h5 \
    --control-mode "pd_ee_delta_pose" --sim-backend "physx_cuda" --num_demos $demos --max_episode_steps 150 \
    --total_iters 100000 --log_freq 100 --eval_freq 5000 \
    --exp-name=act-PushT-v1-state-${demos}_rl_demos-$seed \
    --demo_type rl --track
done

# RGB based

for demos in 100; do
  python train_rgbd.py --env-id PickCube-v1 --no_include_depth \
    --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5 \
    --control-mode "pd_ee_delta_pos" --sim-backend "physx_cuda" --num_demos $demos --max_episode_steps 100 --num_eval_envs 100 --no-capture-video \
    --total_iters 30000 --log_freq 100 --eval_freq 5000 \
    --exp-name=act-PickCube-v1-state-${demos}_motionplanning_demos-$seed \
    --demo_type motionplanning --track

  python train_rgbd.py --env-id PushCube-v1 --no_include_depth \
    --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5 \
    --control-mode "pd_ee_delta_pos" --sim-backend "physx_cuda" --num_demos $demos --max_episode_steps 100 --num_eval_envs 100 --no-capture-video \
    --total_iters 30000 --log_freq 100 --eval_freq 5000 \
    --exp-name=act-PushCube-v1-state-${demos}_motionplanning_demos-$seed \
    --demo_type motionplanning --track

  python train_rgbd.py --env-id StackCube-v1 --no_include_depth \
    --demo-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5 \
    --control-mode "pd_ee_delta_pos" --sim-backend "physx_cuda" --num_demos $demos --max_episode_steps 200 --num_eval_envs 100 --no-capture-video \
    --total_iters 30000 --log_freq 100 --eval_freq 5000 \
    --exp-name=act-StackCube-v1-state-${demos}_motionplanning_demos-$seed \
    --demo_type motionplanning --track

  python train_rgbd.py --env-id PegInsertionSide-v1 --no_include_depth \
    --demo-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5 \
    --control-mode "pd_ee_delta_pose" --sim-backend "physx_cpu" --num_demos $demos --max_episode_steps 300 \
    --total_iters 100000 --log_freq 100 --eval_freq 5000 \
    --exp-name=act-PegInsertionSide-v1-state-${demos}_motionplanning_demos-$seed \
    --demo_type motionplanning --track

  python train_rgbd.py --env-id PushT-v1 --no_include_depth \
    --demo-path ~/.maniskill/demos/PushT-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
    --control-mode "pd_ee_delta_pose" --sim-backend "physx_cuda" --num_demos $demos --max_episode_steps 150 \
    --total_iters 100000 --log_freq 100 --eval_freq 5000 \
    --exp-name=act-PushT-v1-rgb-${demos}_rl_demos-$seed \
    --demo_type rl --track
done