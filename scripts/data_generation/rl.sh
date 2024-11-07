# State-based RL/PPO used to learn a policy to then rollout success demonstrations
# Weights for the trained models are on our hugging face dataset: TODO

# to use these commands you need to install torchrl and tensordict. If cudagraphs does not work you can remove that flag
# then go to the examples/baselines/ppo folder


python ppo_fast.py --env_id="PickCube-v1" \
  --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=5_000_000 --eval_freq=100 \
  --save-model --cudagraphs --exp-name="data_generation/PickCube-v1-ppo"

python ppo_fast.py --env_id="PickCube-v1" --evaluate \
  --checkpoint=runs/data_generation/PickCube-v1-ppo/final_ckpt.pt \
  --num_eval_envs=1024 --num-eval-steps=50 --no-capture-video --save-trajectory

python ppo_fast.py --env_id="StackCube-v1" \
  --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=50_000_000 \
  --save-model --cudagraphs --exp-name="data_generation/StackCube-v1-ppo"

python ppo_fast.py --env_id="StackCube-v1" --evaluate \
  --checkpoint=runs/data_generation/StackCube-v1-ppo/final_ckpt.pt \
  --num_eval_envs=1024 --num-eval-steps=50 --no-capture-video --save-trajectory

python ppo_fast.py --env_id="PushT-v1" \
  --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=25_000_000 --num-eval-steps=100 --gamma=0.99 \
  --save-model --cudagraphs --exp-name="data_generation/PushT-v1-ppo"

python ppo_fast.py --env_id="PushT-v1" --evaluate \
  --checkpoint=runs/data_generation/PushT-v1-ppo/final_ckpt.pt \
  --num_eval_envs=1024 --num-eval-steps=100 --no-capture-video --save-trajectory


python ppo_fast.py --env_id="AnymalC-Reach-v1" \
  --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=10_000_000 --num-eval-steps=200 \
  --gamma=0.99 --gae_lambda=0.95 \
  --save-model --cudagraphs --exp-name="data_generation/AnymalC-Reach-v1-ppo"



python train.py --env-id StackCube-v1 \
  --demo-path ../../../demos/StackCube-v1/rl/trajectory.state.pd_joint_delta_pos.cuda.h5 \
  --control-mode "pd_joint_delta_pos" --sim-backend "gpu" --num-demos 100 --max_episode_steps 50 \
  --total_iters 30000 --num_eval_envs 100