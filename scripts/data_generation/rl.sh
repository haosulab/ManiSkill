# State-based RL/PPO used to learn a policy to then rollout success demonstrations for different controller modes
# Weights for the trained models and pre-generated demos are on our hugging face dataset: https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations

# to use these commands you need to install torchrl and tensordict. If cudagraphs does not work you can remove that flag
# then go to the examples/baselines/ppo folder and run the commands there
# Then run the following commands to preprocess the demos
# python scripts/data_generation/process_rl_trajectories.py --runs_path examples/baselines/ppo/runs/data_generation/ --out-dir ~/.maniskill/demos/

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
for control_mode in "pd_joint_delta_pos" "pd_ee_delta_pos" "pd_ee_delta_pose"; do
  python ppo_fast.py --env_id="RollBall-v1" \
    --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=20_000_000 --num-eval-steps=80 --gamma=0.95 \
    --save-model --cudagraphs --exp-name="data_generation/RollBall-v1-ppo-${control_mode}" --control-mode ${control_mode}

  python ppo_fast.py --env_id="RollBall-v1" --evaluate --control-mode ${control_mode} \
    --checkpoint=runs/data_generation/RollBall-v1-ppo-${control_mode}/final_ckpt.pt \
    --num_eval_envs=1024 --num-eval-steps=80 --no-capture-video --save-trajectory
done

### PokeCube-v1 ###
for control_mode in "pd_joint_delta_pos" "pd_ee_delta_pos" "pd_ee_delta_pose"; do
  python ppo_fast.py --env_id="PokeCube-v1" \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=20_000_000 --eval_freq=100 \
    --save-model --cudagraphs --exp-name="data_generation/PokeCube-v1-ppo-${control_mode}" --control-mode ${control_mode}

  python ppo_fast.py --env_id="PokeCube-v1" --evaluate --control-mode ${control_mode} \
    --checkpoint=runs/data_generation/PokeCube-v1-ppo-${control_mode}/final_ckpt.pt \
    --num_eval_envs=1024 --num-eval-steps=50 --no-capture-video --save-trajectory
done

### PullCube-v1 ###
for control_mode in "pd_joint_delta_pos" "pd_ee_delta_pos" "pd_ee_delta_pose"; do
  python ppo_fast.py --env_id="PullCube-v1" \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=5_000_000 --eval_freq=100 \
    --save-model --cudagraphs --exp-name="data_generation/PullCube-v1-ppo-${control_mode}" --control-mode ${control_mode}
  python ppo_fast.py --env_id="PullCube-v1" --evaluate --control-mode ${control_mode} \
    --checkpoint=runs/data_generation/PullCube-v1-ppo-${control_mode}/final_ckpt.pt \
    --num_eval_envs=1024 --num-eval-steps=50 --no-capture-video --save-trajectory
done

### LiftPegUpright-v1 ###
for control_mode in "pd_joint_delta_pos" "pd_ee_delta_pose"; do
  python ppo_fast.py --env_id="LiftPegUpright-v1" \
    --num_envs=4096 --num-steps=4 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=8_000_000 --eval_freq=100 \
    --save-model --cudagraphs --exp-name="data_generation/LiftPegUpright-v1-ppo-${control_mode}" --control-mode ${control_mode}

  python ppo_fast.py --env_id="LiftPegUpright-v1" --evaluate --control-mode ${control_mode} \
    --checkpoint=runs/data_generation/LiftPegUpright-v1-ppo-${control_mode}/final_ckpt.pt \
    --num_eval_envs=1024 --num-eval-steps=50 --no-capture-video --save-trajectory
done

### AnymalC-Reach-v1 ###
python ppo_fast.py --env_id="AnymalC-Reach-v1" \
  --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=10_000_000 --num-eval-steps=200 \
  --gamma=0.99 --gae_lambda=0.95 \
  --save-model --cudagraphs --exp-name="data_generation/AnymalC-Reach-v1-ppo-pd_joint_delta_pos"

python ppo_fast.py --env_id="AnymalC-Reach-v1" --evaluate \
  --checkpoint=runs/data_generation/AnymalC-Reach-v1-ppo-pd_joint_delta_pos/final_ckpt.pt \
  --num_eval_envs=1024 --num-eval-steps=200 --no-capture-video --save-trajectory

### AnymalC-Spin-v1 ###
python ppo_fast.py --env_id="AnymalC-Spin-v1" \
  --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=10_000_000 --num-eval-steps=200 \
  --gamma=0.99 --gae_lambda=0.95 \
  --save-model --cudagraphs --exp-name="data_generation/AnymalC-Spin-v1-ppo-pd_joint_delta_pos"

# task has no success so no demos for now

### PegInsertionSide-v1 ###
for control_mode in "pd_ee_delta_pose"; do
  python ppo_fast.py --env_id="PegInsertionSide-v1" \
    --num_envs=1024 --num-steps=100 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=100_000_000 --num-eval-steps=100 --gamma=0.97 --gae_lambda=0.95 \
    --save-model --cudagraphs --exp-name="data_generation/PegInsertionSide-v1-ppo-${control_mode}" --control-mode ${control_mode}
done

### TwoRobotPickCube-v1 ###
for control_mode in "pd_joint_delta_pos"; do
  python ppo_fast.py --env_id="TwoRobotPickCube-v1" \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=35_000_000 --num-steps=100 --num-eval-steps=100 \
    --save-model --cudagraphs --exp-name="data_generation/TwoRobotPickCube-v1-ppo-${control_mode}" --control-mode ${control_mode}

  python ppo_fast.py --env_id="TwoRobotPickCube-v1" --evaluate --control-mode ${control_mode} \
    --checkpoint=runs/data_generation/TwoRobotPickCube-v1-ppo-${control_mode}/final_ckpt.pt \
    --num_eval_envs=1024 --num-eval-steps=100 --no-capture-video --save-trajectory
done

### TwoRobotStackCube-v1 ###
for control_mode in "pd_joint_delta_pos"; do
  python ppo_fast.py --env_id="TwoRobotStackCube-v1" \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 --num-steps=100 --num-eval-steps=100 \
    --save-model --cudagraphs --exp-name="data_generation/TwoRobotStackCube-v1-ppo-${control_mode}" --control-mode ${control_mode}

  python ppo_fast.py --env_id="TwoRobotStackCube-v1" --evaluate --control-mode ${control_mode} \
    --checkpoint=runs/data_generation/TwoRobotStackCube-v1-ppo-${control_mode}/final_ckpt.pt \
    --num_eval_envs=1024 --num-eval-steps=100 --no-capture-video --save-trajectory
done

### UnitreeG1PlaceAppleInBowl-v1 ###
# num-steps=32 can be optimized down probably
for control_mode in "pd_joint_delta_pos"; do
  python ppo_fast.py --env_id="UnitreeG1PlaceAppleInBowl-v1" \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 --num-steps=32 --num-eval-steps=100 \
    --save-model --cudagraphs --exp-name="data_generation/UnitreeG1PlaceAppleInBowl-v1-ppo-${control_mode}" --control-mode ${control_mode}
done

### UnitreeG1TransportBox-v1 ###
for control_mode in "pd_joint_delta_pos"; do
  python ppo_fast.py --env_id="UnitreeG1TransportBox-v1" \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 --num-steps=32 --num-eval-steps=100 \
    --save-model --cudagraphs --exp-name="data_generation/UnitreeG1TransportBox-v1-ppo-${control_mode}" --control-mode ${control_mode}
done
