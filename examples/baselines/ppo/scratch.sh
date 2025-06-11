# classic pick cube
for seed in ${seeds[@]}
do
  python ppo_fast.py --env_id="SO100GraspCube-v1" --seed=${seed} \
    --num_envs=4096 --num-steps=8 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --cudagraphs --exp-name="scratch/ppo-SO100GraspCube-v1-state-${seed}-walltime_efficient" \
    --track --wandb_project_name "SO100-ManiSkill"
done

python ppo_rgb.py --env_id="SO100GraspCube-v1" --seed=${seed} \
    --num_envs=1024 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps=50_000_000 \
    --num_eval_envs=16 \
    --cached_resets \
    --exp-name="scratch/ppo-SO100GraspCube-v1-rgb-${seed}-walltime_efficient" \
    --track --wandb_project_name "SO100-ManiSkill"



# this time we use 0.05 arm delta and 0.2 max gripper delta
python ppo_rgb.py --env_id="SO100GraspCube-v1" --seed=${seed} \
  --num_envs=1024 --num-steps=20 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=50_000_000 \
  --num_eval_envs=16 --num-eval-steps=80 \
  --cached_resets \
  --exp-name="scratch/ppo-SO100GraspCube-v1-rgb-${seed}-walltime_efficient-80steps-rewardv3-realbg-targetcontrolmode" \
  --track --wandb_project_name "SO100-ManiSkill" --gamma=0.9

python ppo_rgb.py --env_id="SO100GraspCube-v1" --seed=${seed} \
  --num_envs=1024 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=50_000_000 \
  --num_eval_envs=16 --num-eval-steps=64 \
  --exp-name="scratch/ppo-SO100GraspCube-v1-rgb-${seed}-walltime_efficient-64steps-rewardv3-realbg-targetcontrolmode2" \
  --track --wandb_project_name "SO100-ManiSkill" --gamma=0.95 --gae_lambda=0.9

## reward v4: no success reward exactly, just get as close to goal qpos as possible

python ppo_rgb.py --env_id="SO100GraspCube-v1" --seed=${seed} \
  --num_envs=1024 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=50_000_000 \
  --num_eval_envs=16 --num-eval-steps=64 \
  --exp-name="scratch/ppo-SO100GraspCube-v1-rgb-${seed}-walltime_efficient-64steps-rewardv4-realbg-targetcontrolmode2" \
  --track --wandb_project_name "SO100-ManiSkill" --gamma=0.9

def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
    tcp_to_obj_dist = torch.linalg.norm(
        self.cube.pose.p - self.agent.tcp_pose.p, axis=1
    )
    reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
    reward = reaching_reward

    is_grasped = info["is_grasped"]
    reward += is_grasped
    place_reward = torch.exp(-2 * info["distance_to_rest_qpos"])
    reward += place_reward * is_grasped  # * info["cube_lifted"]
    return reward

def compute_normalized_dense_reward(
    self, obs: Any, action: torch.Tensor, info: Dict
):
    return self.compute_dense_reward(obs=obs, action=action, info=info) / 3



python ppo_rgb.py --env_id="SO100GraspCube-v1" --seed=${seed} \
  --num_envs=1024 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=50_000_000 \
  --num_eval_envs=16 --num-eval-steps=64 \
  --exp-name="scratch/ppo-SO100GraspCube-v1-rgb-${seed}-walltime_efficient-64steps-rewardv4-realbg-targetcontrolmode3-nopartialreset-drlighting" \
  --track --wandb_project_name "SO100-ManiSkill" --gamma=0.9 --no-partial-reset


python ppo_rgb.py --env_id="SO100GraspCube-v1" --seed=${seed} \
  --num_envs=1024 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=100_000_000 \
  --num_eval_envs=16 --num-eval-steps=64 \
  --exp-name="scratch/ppo-SO100GraspCube-v1-rgb-${seed}-walltime_efficient-64steps-rewardv4-realbg-targetcontrolmode3-nopartialreset-drlighting-spawnbox0.3" \
  --track --wandb_project_name "SO100-ManiSkill" --gamma=0.9 --no-partial-reset

python ppo_rgb.py --env_id="SO100GraspCube-v1" --seed=${seed} \
  --num_envs=1024 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=100_000_000 \
  --num_eval_envs=16 --num-eval-steps=64 \
  --exp-name="scratch/ppo-SO100GraspCube-v1-rgb-${seed}-walltime_efficient-64steps-rewardv4-realbg-targetcontrolmode3-nopartialreset-drlighting-spawnbox0.2" \
  --track --wandb_project_name "SO100-ManiSkill" --gamma=0.9 --no-partial-reset