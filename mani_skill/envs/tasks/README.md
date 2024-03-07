# All new ManiSkill tasks

To add a new task, create either a standalone python file your_task_name.py or folder (useful if you use some custom assets). We recommend using the mani_skill/envs/template.py file to get started and filling out the code in there and following the comments.

Moreover, each task is required to come along with a "Task Sheet" which quickly describes some core components of the task, namely
- Randomization: how is the task randomized upon each environment reset?
- Success Conditions: when is the task considered solved?
- Visualization: Link to a gif / video of the task being solved


Difficulty rankings (training on 4090 GPU): 
1 - PPO takes < 2 minutes to train?
2 - PPO takes < 5 minutes
3 - PPO takes < 10 minutes
4 - PPO takes < 1 hour
5 - PPO takes < 12 hours
6 - ?
7 - Unsolved

optimal solution horizon: average time to solve the task when uniformally performing environment resets with a close to optimal policy 

Verified tasks that PPO can solve

env_id, task code, max_episode_steps, optimal solution_horizon, difficulty
PushCube-v1, push_cube.py, 50, ~12, 1
PickCube-v1, pick_cube.py, 50, ~15, 3?
StackCube-v1, stack_cube.py, 50, ~15, 4?
TwoRobotStackCube-v1, two_robot_stack_cube.py, 100, ~20