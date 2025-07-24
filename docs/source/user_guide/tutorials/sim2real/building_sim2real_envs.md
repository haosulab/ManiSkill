# Sim2Real Env Design

WIP

<!-- ## 1 | Create the Sim Environment for Sim2Real

This is possibly the hardest part because not everything in the real world can be simulated to begin with. To limit the scope we will only focus on rigid-body physics (which do cover a significant number of common robotics tasks). Moreover for this sim2real tutorial we will be using a green-screening approach to mitigate part of the vision gap between simulation and real world although for more advanced use cases you might not need it.

To make a simulation environment compatible with `Sim2RealEnv` class and eventually have some success with sim2real transfer you need to ensure the following:

- The simulation environment's `_get_obs_extra` and `_get_obs_agent` function does not use any simulation only features, only features the real-world setup has reliable access to and is sufficiently accurate.
- There is a robot class that implements the {py:class}`mani_skill.agents.base_agent.BaseAgent` class, simulating your real robot. Tutorial for importing custom robots can be found [here](./custom_robots.md). Otherwise you can use one of the existing robot implementations provided by ManiSkill and perform system ID to align simulation robot parameters with your real robot.

The `Sim2RealEnv` will re-use the simulation environment's `_get_obs_extra` and `_get_obs_agent` functions to construct the real environment's observations so that you don't have to.

We note a few common recommendations with respect to these functions:
- The default `_get_obs_agent` function will include the robot's joint position (`qpos`), joint velocity (`qvel`), and any controller state (such as joint targets in controllers that use targets). `qpos` on any robot hardware is generally available and accurate. `qvel` is not always available on some real hardware and estimating these values might be too inaccurate for sim2real transfer sometimes. This tutorial purposely removes `qvel` from the observation as the tutorial uses the Koch robot which does not have accurate joint velocity measurements at the moment.

If you haven't already first follow the [custom tasks tutorial](../custom_tasks/intro.md) to learn how to create a simulation environment, which documents how to load the robot, load objects, setup cameras and more. Once you know how to create a basic environment, go ahead and make a cube picking task similar to the example environment below. The objective is to train a robot to grasp a cube and lift it up. You will note that we inherit from `BaseDigitalTwinEnv` instead of the usual `BaseEnv` class for simulation. This is so we can use the green-screeening functionalities provided by `BaseDigitalTwinEnv` already. The example code below is in the ManiSkill package at [`mani_skill.envs.tasks.digital_twins.tabletop.koch_pickcube`](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/digital_twins/tabletop/koch_pickcube.py) and is heavily annotated to explain most of the lines of code (we do recommend reading the file directly instead of through the documentation here). You can skip the code with respect to the reward function for now as we will cover that in the [last section](#4--reward-function-design) of this tutorial.

:::{dropdown} Example simulation environment code
:::{literalinclude} ../../../../../mani_skill/envs/tasks/digital_twins/so100_arm/grasp_cube.py
    :language: python
    :linenos:
:::


Once you have created your simulation environment, you can first test it by running it with RGB observations as so

```python
import gymnasium as gym
import your_env_file
sim_env = gym.make("MyEnvironment-v1", obs_mode="rgb")
sim_obs, _ = sim_env.reset()
print(sim_obs.keys())
```

TODO photo



## 2 | Setup a Real Agent

For the `Sim2RealEnv` class to work, you need to create a real robot agent that inherits from {py:class}`mani_skill.agents.base_real_agent.BaseRealAgent`. This class essentially is a wrapper around your real robot's hardware interfacing code. For this tutorial we show an example of how to use LeRobot to create the `BaseRealAgent`. The tutorial will be using the low-cost [Koch Robot](https://github.com/jess-moss/koch-v1-1) as an example. You have the following functions to implement:


The goal of the `BaseRealAgent` is to provide a real version of the simulated `BaseAgent` class. This also means that all the implementations are expected to return batched (even if its just a single element) torch tensors in line with the rest of ManiSkill simulation code. An example minimal implementation for the Koch robot is shown below which implements all of the required functions

- `start`: Starts the robot and any real sensors
- `stop`: Stops the robot and any real sensors
- `set_target_qpos`: Sets the target joint positions of the robot
- `set_target_qvel`: Sets the target joint velocities of the robot (optional)
- `capture_sensor_data`: Captures the sensor data of the robot without blocking the main thread
- `get_sensor_data`: Returns the sensor data fetched by the capture function above.
- `get_qpos`: Returns the current joint positions of the robot
- `get_qvel`: Returns the current joint velocities of the robot (optional)

See {py:class}`mani_skill.agents.base_real_agent.BaseRealAgent` for docstrings with more details on the typings and expected behaviors.

:::{literalinclude} ../../../../../mani_skill/agents/robots/lerobot/manipulator.py
    :language: python
    :linenos:
:::

Note that when creating your real agent the `Sim2RealEnv` will provide it access to the simulation equivalent of that real robot so that when your Sim2Real environment fetches some property such as `self.agent.tcp.pose.raw_pose` (the pose of the tool center point of the robot which is common amongst robots and is provided in the Koch robot example) that normally is simulation only, it will be able to do so in the real environment as well. This is because the `Sim2RealEnv` will automatically update the simulation robot's joint positions to whatever the real robot is at, ensuring we always have a simulation digital twin of the real robot, enabling direct use of features that can be computed based on robot features only.

:::{dropdown} Further details on why a simulation is used when deploying a real robot
When doing real world deployments there is no need to create a simulation environment. However, from past research and experience with sim2real transfer, we find that it is very important to align as much as possible between the simulation and real world, which even includes some possibly trivial aspects like forward/inverse kinematics. We had past experiments where we find using the hardware provided inverse kinematics code was not a good idea for sim2real transfer because it is just slightly different compared to the simulation inverse kinematics code. Likewise for forward kinematics. Thus we recommend instantiating a simulation environment (the strategy used with the `Sim2RealEnv` class) which would have a simulation version of the real robot which can have its pose/joint positions set to whatever the real robot is at and now you get access to all the same simulation data in the real world with much less sim2real gap (any gap is now due to sensor noise, not minor code differences).
:::

## 3 | System ID of Real and Simulated Robot


## 4 | Reward Function Design

This is a very important step for RL based sim2real transfer and also one of the hardest aspects to get right. A bad reward function can lead to reward hacking where the RL trained policy might solve the task in an unintentional/unsafe manner (e.g. going too fast, unstable grasps etc.). There is interesting research that can bypass the need to carefully design reward functions by leveraging sparse-reward RL and demonstration data but that is out of the scope of this tutorial. This tutorial will focus on reward function design for a cube picking task that can then be optimized by RL algorithms.

When it comes to designing a reward function for manipulation tasks a common approach is to first decompose the task into stages. Take a look at what the starting state of the task is and what the ending state should be. In this cube picking task, the starting state is the robot arm in a neutral pose with the cube not grasped. The ending state is the robot arm grasping the cube stably in the air. A natural decomposition is the following

1. Move the robot's grippers to be around the cube to prepare for grasping
2. Grasp the cube
3. Lift the cube up
4. Move the cube to some target location.

 -->
