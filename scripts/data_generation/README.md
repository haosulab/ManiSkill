# Data Generation Scripts

The code/scripts in this folder are used to generate all the demonstration datasets for the ManiSkill Benchmark as well as generate the standard state-based and vision-based demonstration datasets for imitation learning baselines.

The tasks that have demonstrations are documented here with the "has demonstrations" tag: https://maniskill.readthedocs.io/en/latest/tasks/index.html

- `replay_for_il_baselines.sh`: This script is used to generate the state-based and vision-based demonstration datasets for imitation learning baselines using the demonstration datasets uploaded to the ManiSkill HuggingFace dataset: https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations
- `rl.sh`: This script is used for using reinforcement learning to learn a policy from dense rewards to then rollout success demonstrations for different controller modes
- `motionplanning.sh`: This script is used for generating the motion planning demonstrations for various tasks that have predefined motion planning solutions