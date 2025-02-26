# Baseline scripts

# state based baselines
seed=1
for demos in 100; do
  python train.py --env-id PickCube-v1 \
    --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
    --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 100 \
    --total_iters 30000 \
    --exp-name diffusion_policy-PickCube-v1-state-${demos}_motionplanning_demos-${seed} \
    --demo_type=motionplanning --track

  python train.py --env-id PushCube-v1 \
    --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
    --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 100 \
    --total_iters 30000 \
    --exp-name diffusion_policy-PushCube-v1-state-${demos}_motionplanning_demos-${seed} \
    --demo_type=motionplanning --track

  python train.py --env-id PushT-v1 \
    --demo-path ~/.maniskill/demos/PushT-v1/rl/trajectory.state.pd_ee_delta_pose.physx_cuda.h5 \
    --control-mode "pd_ee_delta_pose" --sim-backend "physx_cuda" --num-demos ${demos} --max_episode_steps 150 --num_eval_envs 100 \
    --total_iters 50000 --act_horizon 1 \
    --exp-name diffusion_policy-PushT-v1-state-${demos}_rl_demos-${seed} --no_capture_video \
    --demo_type=rl --track

  python train.py --env-id StackCube-v1 \
    --demo-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
    --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 200 \
    --total_iters 30000 \
    --exp-name diffusion_policy-StackCube-v1-state-${demos}_motionplanning_demos-${seed} \
    --demo_type=motionplanning --track

  python train.py --env-id PegInsertionSide-v1 \
    --demo-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.h5 \
    --control-mode "pd_ee_delta_pose" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 300 \
    --total_iters 100000 \
    --exp-name diffusion_policy-PegInsertionSide-v1-state-${demos}_motionplanning_demos-${seed} \
    --demo_type=motionplanning --track
done

# rgb based baselines
for demos in 100; do
  python train_rgbd.py --env-id PickCube-v1 \
    --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5 \
    --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 100 \
    --total_iters 30000 --obs-mode "rgb" \
    --exp-name diffusion_policy-PickCube-v1-rgb-${demos}_motionplanning_demos-${seed} \
    --demo_type=motionplanning --track

  python train_rgbd.py --env-id PushCube-v1 \
    --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5 \
    --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 100 \
    --total_iters 30000 --obs-mode "rgb" \
    --exp-name diffusion_policy-PushCube-v1-rgb-${demos}_motionplanning_demos-${seed} \
    --demo_type=motionplanning --track

  python train_rgbd.py --env-id StackCube-v1 \
    --demo-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5 \
    --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 200 \
    --total_iters 100000 --obs-mode "rgb" \
    --exp-name diffusion_policy-StackCube-v1-rgb-${demos}_motionplanning_demos-${seed} \
    --demo_type=motionplanning --track
    
  python train_rgbd.py --env-id DrawTriangle-v1 \
    --demo-path ~/.maniskill/demos/DrawTriangle-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5 \
    --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 300 \
    --total_iters 100000 --obs-mode "rgb" --batch_size 128 \
    --exp-name diffusion_policy-DrawTriangle-v1-rgb-${demos}_motionplanning_demos-${seed} \
    --demo_type=motionplanning --track
done