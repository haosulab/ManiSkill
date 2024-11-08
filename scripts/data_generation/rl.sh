# State-based RL/PPO used to learn a policy to then rollout success demonstrations for different controller modes
# Weights for the trained models are on our hugging face dataset: TODO

# to use these commands you need to install torchrl and tensordict. If cudagraphs does not work you can remove that flag
# then go to the examples/baselines/ppo folder


### PushCube-v1 ###
for control_mode in "pd_joint_delta_pos" "pd_ee_delta_pos" "pd_ee_delta_pose"; do
  python ppo_fast.py --env_id="PushCube-v1" \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=5_000_000 --eval_freq=100 \
    --save-model --cudagraphs --exp-name="data_generation/PushCube-v1-ppo-${control_mode}" --control-mode ${control_mode}
  
  python ppo_fast.py --env_id="PushCube-v1" --evaluate --control-mode ${control_mode} \
    --checkpoint=runs/data_generation/PushCube-v1-ppo-${control_mode}/final_ckpt.pt \
    --num_eval_envs=1024 --num-eval-steps=50 --no-capture-video --save-trajectory
done


### PickCube-v1 ###
for control_mode in "pd_joint_delta_pos" "pd_ee_delta_pos" "pd_ee_delta_pose"; do
  python ppo_fast.py --env_id="PickCube-v1" \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=5_000_000 --eval_freq=100 \
    --save-model --cudagraphs --exp-name="data_generation/PickCube-v1-ppo-${control_mode}" --control-mode ${control_mode}

  python ppo_fast.py --env_id="PickCube-v1" --evaluate --control-mode ${control_mode} \
    --checkpoint=runs/data_generation/PickCube-v1-ppo-${control_mode}/final_ckpt.pt \
    --num_eval_envs=1024 --num-eval-steps=50 --no-capture-video --save-trajectory
done

### StackCube-v1 ###
for control_mode in "pd_joint_delta_pos" "pd_ee_delta_pos" "pd_ee_delta_pose"; do
  python ppo_fast.py --env_id="StackCube-v1" \
    --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --save-model --cudagraphs --exp-name="data_generation/StackCube-v1-ppo-${control_mode}" --control-mode ${control_mode}

  python ppo_fast.py --env_id="StackCube-v1" --evaluate --control-mode ${control_mode} \
    --checkpoint=runs/data_generation/StackCube-v1-ppo-${control_mode}/final_ckpt.pt \
    --num_eval_envs=1024 --num-eval-steps=50 --no-capture-video --save-trajectory
done

### PushT-v1 ###
for control_mode in "pd_joint_delta_pos" "pd_ee_delta_pos" "pd_ee_delta_pose"; do
  python ppo_fast.py --env_id="PushT-v1" \
    --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=25_000_000 --num-eval-steps=100 --gamma=0.99 \
    --save-model --cudagraphs --exp-name="data_generation/PushT-v1-ppo-${control_mode}" --control-mode ${control_mode}
  python ppo_fast.py --env_id="PushT-v1" --evaluate --control-mode ${control_mode} \
    --checkpoint=runs/data_generation/PushT-v1-ppo-${control_mode}/final_ckpt.pt \
    --num_eval_envs=1024 --num-eval-steps=100 --no-capture-video --save-trajectory
done

### RollBall-v1 ###
python ppo_fast.py --env_id="RollBall-v1" \
  --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=20_000_000 --num-eval-steps=80 --gamma=0.95 \
  --save-model --cudagraphs --exp-name="data_generation/RollBall-v1-ppo"

python ppo_fast.py --env_id="RollBall-v1" --evaluate \
  --checkpoint=runs/data_generation/RollBall-v1-ppo/final_ckpt.pt \
  --num_eval_envs=1024 --num-eval-steps=80 --no-capture-video --save-trajectory

### PokeCube-v1 ###
python ppo_fast.py --env_id="PokeCube-v1" \
  --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=15_000_000 --eval_freq=100 \
  --save-model --cudagraphs --exp-name="data_generation/PokeCube-v1-ppo"

python ppo_fast.py --env_id="PokeCube-v1" --evaluate \
  --checkpoint=runs/data_generation/PokeCube-v1-ppo/final_ckpt.pt \
  --num_eval_envs=1024 --num-eval-steps=50 --no-capture-video --save-trajectory

### PullCube-v1 ###
python ppo_fast.py --env_id="PullCube-v1" \
  --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=5_000_000 --eval_freq=100 \
  --save-model --cudagraphs --exp-name="data_generation/PullCube-v1-ppo"

python ppo_fast.py --env_id="PullCube-v1" --evaluate \
  --checkpoint=runs/data_generation/PullCube-v1-ppo/final_ckpt.pt \
  --num_eval_envs=1024 --num-eval-steps=50 --no-capture-video --save-trajectory

### LiftPegUpright-v1 ###
python ppo_fast.py --env_id="LiftPegUpright-v1" \
  --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=8_000_000 --eval_freq=100 \
  --save-model --cudagraphs --exp-name="data_generation/LiftPegUpright-v1-ppo"

python ppo_fast.py --env_id="LiftPegUpright-v1" --evaluate \
  --checkpoint=runs/data_generation/LiftPegUpright-v1-ppo/final_ckpt.pt \
  --num_eval_envs=1024 --num-eval-steps=50 --no-capture-video --save-trajectory

### AnymalC-Reach-v1 ###
python ppo_fast.py --env_id="AnymalC-Reach-v1" \
  --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=10_000_000 --num-eval-steps=200 \
  --gamma=0.99 --gae_lambda=0.95 \
  --save-model --cudagraphs --exp-name="data_generation/AnymalC-Reach-v1-ppo"

python ppo_fast.py --env_id="AnymalC-Reach-v1" --evaluate \
  --checkpoint=runs/data_generation/AnymalC-Reach-v1-ppo/final_ckpt.pt \
  --num_eval_envs=1024 --num-eval-steps=200 --no-capture-video --save-trajectory

### AnymalC-Spin-v1 ###
python ppo_fast.py --env_id="AnymalC-Spin-v1" \
  --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=10_000_000 --num-eval-steps=200 \
  --gamma=0.99 --gae_lambda=0.95 \
  --save-model --cudagraphs --exp-name="data_generation/AnymalC-Spin-v1-ppo"

# task has no success so no demos for now






python train.py --env-id StackCube-v1 \
  --demo-path ../../../demos/StackCube-v1/rl/trajectory.state.pd_ee_delta_pos.cuda.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "gpu"  --max_episode_steps 50 \
  --total_iters 100000 --num_eval_envs 16

python bc.py --env-id "StackCube-v1" \
  --demo-path ../../../demos/StackCube-v1/rl/trajectory.state.pd_ee_delta_pos.cuda.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "gpu" --max-episode-steps 50 \
  --total-iters 10000 --num_eval_envs 16