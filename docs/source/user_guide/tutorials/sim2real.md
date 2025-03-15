# Sim2Real

For sim2real one typically needs to align dynamics and visual data. ManiSkill provides a few utilities to help minimize the amount of extra code you need to write and streamline the process. A recommended pre-requisite to making sim2real environments in this tutorial is to first learn how to create simulation tasks in the [custom tasks tutorial](./custom_tasks/intro.md).

We first describe at a high-level some of the features of the {py:class}`mani_skill.envs.sim2real_env.Sim2RealEnv` class that we provide that helps streamline the process of creating sim2real environments. Then the full tutorial will give examples and step-by-step instructions on how to make your own Sim2Real environments using the highly accessible / low-cost [LeRobot](https://github.com/huggingface/lerobot) system for easy robot/sensor setups. Coming soon will also include simple RGB-based sim2real deployment of policies trained with RL entirely in simulation (a demo showcase of that is [in the demo gallery](../demos/gallery.md#vision-based-zero-shot-sim2real-manipulation) if you are interested).

## High Level Overview

The goal of the `Sim2RealEnv` class is to avoid manually writing too much code to ensure aspects like controllers code or sensor image resolution/shapes are aligned with the simulation. The `Sim2RealEnv` class inspects a given simulation environment and automatically tries to setup the real environment to be as similar as possible. It will also perform some automated checks to ensure that the real environment's robot and sensors are correctly configured. This will ensure that the real environment uses the same exact action space and robot controller as the simulation environment, and ensure the observations are the same shape and order (it does not guarantee real-world images match the lower fidelity of simulation images). Moreover the `Sim2RealEnv` class will also follow the Gymnasium interface so it can be used similar to a simulation environment. Code shown hides some of the real robot code setup for brevity, but will be explained in full at the start of the tutorial (next section).


```python
import gymnasium as gym
from mani_skill.envs.sim2real_env import Sim2RealEnv
sim_env = gym.make("YourEnvironment", obs_mode="rgb")
# setup the real robot and sensors/cameras
real_agent = LeRobotAgent(**config)
# create a real world interface based on the sim env and real robot
real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent) 
# assuming sim env does not generate any observations 
# that are not accessible in the real world
# the two observations should be identical
sim_obs, _ = sim_env.reset(seed=0)
# real resets by default will prompt user to reset the real world first before continuing
# the robot by default will reset to whatever joint position the sim env samples
real_obs, _ = real_env.reset(seed=0)
done = False
while not done:
    action = real_env.action_space.sample()
    real_obs, _, terminated, truncated, info = real_env.step(action)
    done = terminated or truncated
sim_env.close()
real_env.close() # or real_agent.stop()
```

For those familiar with Gymnasium/Reinforcement Learning (RL), we also support simulation wrappers on real environments. In order to re-use the wrappers you may have used for training in simulation, you simply apply those wrappers to the `sim_env` before passing it into the `Sim2RealEnv` constructor. A common wrapper is the [FlattenRGBDObservationWrapper](../wrappers/flatten.md#flatten-rgbd-observations) which flattens the observation space to a dictionary with a "state" key and a "rgb" and/or "depth" key. The example below will show how to apply the wrapper and then create a real environment interface and print the observation data/shapes.

```python
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
sim_env = gym.make("YourEnvironment", obs_mode="rgb")
real_agent = LeRobotAgent(**config)
sim_env = FlattenRGBDObservationWrapper(sim_env)
real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent)

sim_obs, _ = sim_env.reset()
real_obs, _ = real_env.reset()
for k in sim_obs.keys():
    print(
        f"{k}: sim_obs shape: {sim_obs[k].shape}, real_obs shape: {real_obs[k].shape}"
    )
# state: sim_obs shape: torch.Size([1, 13]), real_obs shape: torch.Size([1, 13])
# rgb: sim_obs shape: torch.Size([1, 128, 128, 3]), real_obs shape: torch.Size([1, 128, 128, 3])
```

Like the rest of ManiSkill and sim environments, all Sim2Real environments will output batched data by default.

:::{note}
While this may streamline the process of creating sim2real environment interfaces for robot policy deployment, it is quite simple and requires instantiating one CPU simulated environment in order to do the automatic checks and configuration of the controller and sensors. Moreover it is possible there is overhead induced by inefficiencies as this general code cannot easily account for features unique to some real-world setups. We further tried to make it such that most gym wrappers would be compatible with the real environment, but if you run into issues please let us know.

If you need heavy customizations we provide some recommendations in later sections of the tutorial.
:::

## 1 | Create the Sim Environment for Sim2Real

This is possibly the hardest part because not everything in the real world can be simulated to begin with. To limit the scope we will only focus on rigid-body physics (which do cover a significant number of common robotics tasks).

To make a simulation environment compatible with `Sim2RealEnv` class and eventually have some success with sim2real transfer you need to ensure the following:

- The simulation environment's `_get_obs_extra` and `_get_obs_agent` function does not use any simulation only features, only features the real-world setup has reliable access to and is sufficiently accurate.

The `Sim2RealEnv` will re-use the simulation environment's `_get_obs_extra` and `_get_obs_agent` functions to construct the real environment's observations so that you don't have to.

We note a few common recommendations with respect to these functions:
- The default `_get_obs_agent` function will include the robot's joint position (`qpos`), joint velocity (`qvel`), and any controller state (such as joint targets in controllers that use targets). `qpos` on any robot hardware is generally available and accurate. `qvel` is not always available on some real hardware and estimating these values might be too inaccurate for sim2real transfer sometimes.
<!-- - Often users will include end-effector poses (or tool-center-point tcp poses) in the observation via the `_get_obs_extra` function. The Sim2RealEnv will automatically update the simulation environment's robot joint positions to whatever the real robot is at so this kind of data is correct. -->
<!-- TODO (stao): tcp.pose is supportable but its finnicky because agent.tcp is a property of the simulation agent. Would need a KochRealAgent essentially. But i guess each robot setup needs a seperate real agent class anyway? -->

## 2 | Setup a Real Agent








## Dynamics Alignment

If you are looking to 


## Recording Real World Episodes

TODO: use record episode still

## Applying Wrappers in Sim to Real Environments

Often times during RL / IL workflows you will have various useful environment wrappers to modify the aspects such as observations (e.g. to combine rgb and depth images) or actions (e.g. to apply an action sequence). Whatever wrappers you use for the simulation environment can also be applied to the real environment by simply appyling them to the sim environment before passing it into the `RealEnv` constructor.

```python
from mani_skill.envs.real_env import RealEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

wrappers = [FlattenRGBDObservationWrapper]
sim_env = gym.make("YourEnvironment", obs_mode="rgb")
for wrapper in wrappers:
    sim_env = wrapper(sim_env)
real_env = RealEnv(sim_env=sim_env, agent=YourCustomRealRobotClass)
```


## Customizing Real Environments More

By default the `RealEnv` class will infer and setup the observation and controllers as best as possible. However it is possible you might have specific use-cases such as custom real world controller code different from the simulation controller or modifications to the observation space.

You can either write your own code or create a class that inherits from `RealEnv` and overrides the behaviors as needed. To change observation data just change the `get_obs` function and to change the controller/actions just override the `_step_action` function.


```python
class MyCustomRealEnv(RealEnv):
    def _step_action(self, action):
        # custom real world controller code
        pass

    def get_obs(self, info):
        # change the observation space
        return super().get_obs(info)
```



