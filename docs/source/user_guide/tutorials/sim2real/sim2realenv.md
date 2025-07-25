# Sim2RealEnv

The `Sim2RealEnv` class is a gymnasium interface for the real world robot.

The goal of the `Sim2RealEnv` class is to avoid manually writing excessive code while ensuring aspects like controller code or sensor image resolution/shapes are aligned with the simulation. The `Sim2RealEnv` class inspects a given simulation environment and automatically tries to setup the real environment to be as similar as possible. It will also perform some automated checks to ensure that the real environment's robot and sensors are correctly configured. This will ensure that the real environment uses the same exact action space and robot controller as the simulation environment, and ensure the observations are the same shape and order (it does not guarantee real-world images match the fidelity of simulation images). Moreover the `Sim2RealEnv` class will also follow the Gymnasium interface so it can be used similar to a simulation environment. Code shown hides some of the real robot code setup for brevity.


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
real_obs, _ = real_env.reset(seed=0)

# real resets by default will prompt user to reset the real world first before continuing
# the robot by default will reset to whatever joint position the sim env samples

done = False
while not done:
    action = real_env.action_space.sample()
    real_obs, _, terminated, truncated, info = real_env.step(action)
    done = terminated or truncated
sim_env.close()
real_env.close()
```

For those familiar with Gymnasium/Reinforcement Learning (RL), we also support simulation wrappers on real environments. In order to re-use the wrappers you may have used for RL training in simulation, you simply apply those wrappers to the `sim_env` before passing it into the `Sim2RealEnv` constructor. A common wrapper is the [FlattenRGBDObservationWrapper](../../wrappers/flatten.md#flatten-rgbd-observations) which flattens the observation space to a dictionary with a "state" key and a "rgb" and/or "depth" key. The example below will show how to apply the wrapper and then create a real environment interface and print the observation data/shapes.

```python
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
sim_env = gym.make("YourEnvironment", obs_mode="rgb")
real_agent = LeRobotAgent(**config)
sim_env = FlattenRGBDObservationWrapper(sim_env)
real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent)
# real_env inherits all the wrappers sim_env uses

sim_obs, _ = sim_env.reset()
real_obs, _ = real_env.reset()
for k in sim_obs.keys():
    print(
        f"{k}: sim_obs shape: {sim_obs[k].shape}, real_obs shape: {real_obs[k].shape}"
    )
# state: sim_obs shape: torch.Size([1, 13]), real_obs shape: torch.Size([1, 13])
# rgb: sim_obs shape: torch.Size([1, 128, 128, 3]), real_obs shape: torch.Size([1, 128, 128, 3])
```

Like the rest of ManiSkill and simulated environments, all Sim2Real environments will output batched data by default.

:::{note}
While this may streamline the process of creating sim2real environment interfaces for robot policy deployment, it is quite simple and requires instantiating one CPU simulated environment in order to do the automatic checks and configuration of the controller and sensors. Moreover it is possible there is overhead induced by inefficiencies as this general code cannot easily account for features unique to some real-world setups. We further tried to make it such that most gym wrappers would be compatible with the real environment, but if you run into issues please let us know.

If you need heavy customizations we provide some recommendations in later sections of the tutorial.
:::
