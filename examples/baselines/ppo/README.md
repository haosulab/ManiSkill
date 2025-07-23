# PPO for ManiSkill with RGB Observations

This repository contains a Python script that trains a robotic agent to solve complex manipulation tasks in the [ManiSkill2 benchmark](https://maniskill.ai/). It uses **Proximal Policy Optimization (PPO)**, a state-of-the-art reinforcement learning algorithm, and learns directly from RGB camera images and robot joint states.

## 🚀 The Code: `ppo_rgb.py`

The core logic is contained in the `ppo_rgb.py` script. Here's a quick breakdown of its key components:

* **Configuration (`Args` dataclass):** Defines all customizable parameters for the experiment, including environment settings, algorithm hyperparameters, and logging options. These are easily modified via command-line arguments.
* **Network Architecture (`Agent` & `NatureCNN`):** The agent's "brain" is a neural network.
    * A **Convolutional Neural Network (CNN)**, specifically the `NatureCNN`, processes the visual RGB input from the environment's camera.
    * This visual data is combined with the robot's own state information (e.g., joint positions).
    * The combined features are fed into two heads: an **actor** that decides which action to take, and a **critic** that estimates how good the current situation is.
* **PPO Algorithm Flow:** The script follows the standard PPO training loop:
    1.  **Collect Data:** The agent runs in hundreds of parallel environments to collect experience (observations, actions, rewards).
    2.  **Calculate Advantages:** It determines how much better an action was than the baseline average using a technique called Generalized Advantage Estimation (GAE).
    3.  **Update Networks:** It uses the collected data to update the actor and critic networks, improving the agent's decision-making policy over thousands of iterations.
* **ManiSkill Integration:** The code uses ManiSkill's `ManiSkillVectorEnv` for efficient, parallelized simulation and custom wrappers to handle the specific observation and action formats of the benchmark.

---

## ⚙️ How to Run an Experiment

You can start training an agent by running the `ppo_rgb.py` script from your terminal. The command below provides an example of how to train an `xarm6` robot to solve the `PickCube-v1` task.

### Example Command

```bash
python examples/baselines/ppo/ppo_rgb.py \
    --env_id=PickCube-v1 \
    --robot_uids=xarm6_robotiq \
    --control_mode=pd_joint_vel \
    --exp_name=PickCube_xarm6_ppo \
    --num_envs=512 \
    --num_eval_envs=8 \
    --eval_freq=20 \
    --total_timesteps=100_000_000 \
    --num_steps=50 \
    --gamma=0.8 \
    --capture-video \
    --track \
    --wandb_project_name "ManiSkill-RL"
```

This command will initialize the training process, and you'll see output in your terminal. If you use the `--track` flag, you can monitor the agent's learning progress, including success rates and rewards, in real-time on [Weights & Biases](https://wandb.ai/). Videos of the agent's performance will be saved in the `runs/` directory.

### Command-Line Arguments Explained
Here's what the key arguments in the example command do:

* `--env_id`: Specifies the ManiSkill task to solve (e.g., `PickCube-v1`).
* `--robot_uids`: The robot model to use (e.g., `xarm6_robotiq`, `panda`).
* `--control_mode`: The control method for the robot's joints (e.g., `pd_joint_vel` for velocity control).
* `--exp_name`: A custom name for your experiment run.
* `--num_envs`: The number of parallel environments to run for data collection. Higher numbers can speed up training.
* `--total_timesteps`: The total number of environment steps the agent will be trained for.
* `--num_steps`: The number of steps each parallel environment runs before the agent's networks are updated.
* `--gamma`: The discount factor, which determines how much the agent values future rewards. A lower value (like 0.8) makes the agent more "short-sighted."
* `--capture-video`: A flag to enable video recording of the agent's performance during evaluation.
* `--track`: A flag to enable logging with Weights & Biases (W&B).
* `--wandb_project_name`: The name of the W&B project where results will be logged.