# Flattening Data

A suite of common wrappers useful for flattening/transforming data like observations/actions into more useful formats for e.g. Reinforcement Learning or Imitation Learning.

## Flatten Observations

A simple wrapper to flatten a dictionary observation space into a flat array observation space.

```python
import mani_skill.envs
from mani_skill.utils.wrappers import FlattenObservationWrapper
import gymnasium as gym

env = gym.make("PickCube-v1", obs_mode="state_dict")
print(env.observation_space) # is a complex nested dictionary
# Dict('agent': Dict('qpos': Box(-inf, inf, (1, 9), float32), 'qvel': Box(-inf, inf, (1, 9), float32)), 'extra': Dict('is_grasped': Box(False, True, (1,), bool), 'tcp_pose': Box(-inf, inf, (1, 7), float32), 'goal_pos': Box(-inf, inf, (1, 3), float32), 'obj_pose': Box(-inf, inf, (1, 7), float32), 'tcp_to_obj_pos': Box(-inf, inf, (1, 3), float32), 'obj_to_goal_pos': Box(-inf, inf, (1, 3), float32)))
env = FlattenObservationWrapper(env)
print(env.observation_space) # is a flat array now
# Box(-inf, inf, (1, 42), float32)
```

## Flatten Actions

A simple wrapper to flatten a dictionary action space into a flat array action space. Commonly used for multi-agent like environments when you want to control multiple agents/robots together with one action space.

```python
import mani_skill.envs
from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
import gymnasium as gym

env = gym.make("TwoRobotStackCube-v1")
print(env.action_space) # is a dictionary
# Dict('panda-0': Box(-1.0, 1.0, (8,), float32), 'panda-1': Box(-1.0, 1.0, (8,), float32))
env = FlattenActionSpaceWrapper(env)
print(env.action_space) # is a flat array now
# Box(-1.0, 1.0, (16,), float32)
```

## Flatten RGBD Observations

This wrapper concatenates all the RGB and Depth images into a single image with combined channels, and concatenates all state data into a single array so that the observation space becomes a simple dictionary composed of a `state`, `rgb`, and `depth` key.

```python
import mani_skill.envs
from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper
import gymnasium as gym

env = gym.make("PickCube-v1", obs_mode="rgb+depth")
print(env.observation_space) # is a complex dictionary
# Dict('agent': Dict('qpos': Box(-inf, inf, (1, 9), float32), 'qvel': Box(-inf, inf, (1, 9), float32)), 'extra': Dict('is_grasped': Box(False, True, (1,), bool), 'tcp_pose': Box(-inf, inf, (1, 7), float32), 'goal_pos': Box(-inf, inf, (1, 3), float32)), 'sensor_param': Dict('base_camera': Dict('extrinsic_cv': Box(-inf, inf, (1, 3, 4), float32), 'cam2world_gl': Box(-inf, inf, (1, 4, 4), float32), 'intrinsic_cv': Box(-inf, inf, (1, 3, 3), float32))), 'sensor_data': Dict('base_camera': Dict('rgb': Box(0, 255, (1, 128, 128, 3), uint8), 'depth': Box(-32768, 32767, (1, 128, 128, 1), int16))))
env = FlattenRGBDObservationWrapper(env)
print(env.observation_space) # is a much simpler dictionary now
# Dict('state': Box(-inf, inf, (1, 29), float32), 'rgb': Box(-32768, 32767, (1, 128, 128, 3), int16), 'depth': Box(-32768, 32767, (1, 128, 128, 1), int16))
```