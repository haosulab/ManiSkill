# This file is a giant collection of tested example commands for PPO
# Note these are tuned for wall time speed. For official baseline results which run
# more fair comparisons of RL algorithms see the baselines.sh file

### State Based PPO ###
python ppo.py --env_id="PickCube-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=10_000_000
python ppo.py --env_id="StackCube-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=25_000_000
python ppo.py --env_id="PushT-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=25_000_000 --num-steps=100 --num_eval_steps=100 --gamma=0.99
python ppo.py --env_id="PickSingleYCB-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=25_000_000
python ppo.py --env_id="PegInsertionSide-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=250_000_000 --num-steps=100 --num-eval-steps=100
python ppo.py --env_id="TwoRobotPickCube-v1" \
   --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=20_000_000 --num-steps=100 --num-eval-steps=100
python ppo.py --env_id="TwoRobotStackCube-v1" \
   --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=40_000_000 --num-steps=100 --num-eval-steps=100
python ppo.py --env_id="TriFingerRotateCubeLevel0-v1" \
   --num_envs=128 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=50_000_000 --num-steps=250 --num-eval-steps=250
python ppo.py --env_id="TriFingerRotateCubeLevel1-v1" \
   --num_envs=128 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=50_000_000 --num-steps=250 --num-eval-steps=250
python ppo.py --env_id="TriFingerRotateCubeLevel2-v1" \
   --num_envs=128 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=50_000_000 --num-steps=250 --num-eval-steps=250
python ppo.py --env_id="TriFingerRotateCubeLevel3-v1" \
   --num_envs=128 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=50_000_000 --num-steps=250 --num-eval-steps=250
python ppo.py --env_id="TriFingerRotateCubeLevel4-v1" \
   --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=500_000_000 --num-steps=250 --num-eval-steps=250
python ppo.py --env_id="PokeCube-v1" --update_epochs=8 --num_minibatches=32 \
  --num_envs=1024 --total_timesteps=5_000_000 --eval_freq=10 --num-steps=20
python ppo.py --env_id="MS-CartpoleBalance-v1" \
   --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=4_000_000 --num-steps=250 --num-eval-steps=1000 \
   --gamma=0.99 --gae_lambda=0.95 \
   --eval_freq=5

python ppo.py --env_id="MS-CartpoleSwingUp-v1" \
   --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
   --total_timesteps=10_000_000 --num-steps=250 --num-eval-steps=1000 \
   --gamma=0.99 --gae_lambda=0.95 \
   --eval_freq=5
python ppo.py --env_id="MS-AntWalk-v1" --num_envs=2048 --eval_freq=10 \
  --update_epochs=8 --num_minibatches=32 --total_timesteps=20_000_000 \
  --num_eval_steps=1000 --num_steps=200 --gamma=0.97 --ent_coef=1e-3
python ppo.py --env_id="MS-AntRun-v1" --num_envs=2048 --eval_freq=10 \
  --update_epochs=8 --num_minibatches=32 --total_timesteps=20_000_000 \
  --num_eval_steps=1000 --num_steps=200 --gamma=0.97 --ent_coef=1e-3
python ppo.py --env_id="MS-HumanoidStand-v1" --num_envs=2048 --eval_freq=10 \
  --update_epochs=8 --num_minibatches=32 --total_timesteps=40_000_000 \
  --num_eval_steps=1000 --num_steps=200 --gamma=0.95
python ppo.py --env_id="MS-HumanoidWalk-v1" --num_envs=2048 --eval_freq=10 \
  --update_epochs=8 --num_minibatches=32 --total_timesteps=80_000_000 \
  --num_eval_steps=1000 --num_steps=200 --gamma=0.97 --ent_coef=1e-3
python ppo.py --env_id="MS-HumanoidRun-v1" --num_envs=2048 --eval_freq=10 \
  --update_epochs=8 --num_minibatches=32 --total_timesteps=60_000_000 \
  --num_eval_steps=1000 --num_steps=200 --gamma=0.97 --ent_coef=1e-3
python ppo.py --env_id="UnitreeG1PlaceAppleInBowl-v1" \
  --num_envs=512 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=50_000_000 --num-steps=100 --num-eval-steps=100
python ppo.py --env_id="AnymalC-Reach-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=25_000_000 --num-steps=200 --num-eval-steps=200 \
  --gamma=0.99 --gae_lambda=0.95
python ppo.py --env_id="AnymalC-Spin-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=50_000_000 --num-steps=200 --num-eval-steps=200 \
  --gamma=0.99 --gae_lambda=0.95
python ppo.py --env_id="UnitreeGo2-Reach-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=50_000_000 --num-steps=200 --num-eval-steps=200 \
  --gamma=0.99 --gae_lambda=0.95
python ppo.py --env_id="UnitreeH1Stand-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=100_000_000 --num-steps=100 --num-eval-steps=1000 \
  --gamma=0.99 --gae_lambda=0.95
python ppo.py --env_id="UnitreeG1Stand-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=100_000_000 --num-steps=100 --num-eval-steps=1000 \
  --gamma=0.99 --gae_lambda=0.95

python ppo.py --env_id="OpenCabinetDrawer-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=10_000_000 --num-steps=100 --num-eval-steps=100   

python ppo.py --env_id="RollBall-v1" \
  --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=20_000_000 --num-steps=80 --num_eval_steps=80 --gamma=0.95

### RGB Based PPO ###
python ppo_rgb.py --env_id="PushCube-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=8 \
  --total_timesteps=1_000_000 --eval_freq=10 --num-steps=20
python ppo_rgb.py --env_id="PickCube-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=8 \
  --total_timesteps=10_000_000
python ppo_rgb.py --env_id="AnymalC-Reach-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=10_000_000 --num-steps=200 --num-eval-steps=200 \
  --gamma=0.99 --gae_lambda=0.95
python ppo_rgb.py --env_id="PickSingleYCB-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=8 \
  --total_timesteps=10_000_000
python ppo_rgb.py --env_id="PushT-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=8 \
  --total_timesteps=25_000_000 --num-steps=100 --num_eval_steps=100 --gamma=0.99
python ppo_rgb.py --env_id="MS-AntWalk-v1" \
 --num_envs=256 --update_epochs=8 --num_minibatches=32 \
 --total_timesteps=5_000_000 --eval_freq=15 --num_eval_steps=1000 \
 --num_steps=200 --gamma=0.97 --no-include-state --render_mode="rgb_array" \
 --ent_coef=1e-3
python ppo_rgb.py --env_id="MS-AntRun-v1" \
 --num_envs=256 --update_epochs=8 --num_minibatches=32 \
 --total_timesteps=15_000_000 --eval_freq=15 --num_eval_steps=1000 \
 --num_steps=200 --gamma=0.97 --no-include-state --render_mode="rgb_array" \
 --ent_coef=1e-3
python ppo_rgb.py --env_id="MS-HumanoidRun-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=80_000_000 --eval_freq=15 --num_eval_steps=1000 \
  --num_steps=200 --gamma=0.98 --no-include-state --render_mode="rgb_array" \
  --ent_coef=1e-3
