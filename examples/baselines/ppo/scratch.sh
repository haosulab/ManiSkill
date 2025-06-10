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


python ppo_rgb.py --env_id="SO100GraspCube-v1" --seed=${seed} \
  --num_envs=1024 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=50_000_000 \
  --num_eval_envs=16 --num-eval-steps=64 \
  --cached_resets \
  --exp-name="scratch/ppo-SO100GraspCube-v1-rgb-${seed}-walltime_efficient-64steps-rewardv3" \
  --track --wandb_project_name "SO100-ManiSkill"