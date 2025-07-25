# Cached Reset

For some environments where environment resets may be slow/expensive, or during workflows like RL with partial resets where there are frequent small resets (instead of resetting all environments in GPU sim simultaneously), it can be useful to use cached resets.

Cached resets essentially skips the process of calling the environment's reset function and instead loads a previous environment state and observation instead. Loading environment state instead of running environment reset code (the `_initialize_episode` function) can be faster and boost environment FPS.

To use cached resets we provide a simple environment wrapper {py:class}`mani_skill.utils.wrappers.CachedResetWrapper` that can be used as follows


```python
from mani_skill.utils.wrappers import CachedResetWrapper
import gymnasium as gym

env = gym.make("StackCube-v1", num_envs=256)
# upon applying the wrapper below we will by default sample 256 different reset states and the corresponding observations and cache them
env = CachedResetWrapper(env)
# obs is now fetched from a cache, and we initialize the environment with environment state
obs, _ = env.reset()
```

Note that this does not cache geometry/texture details, only environment state. Most ManiSkill environments change geometries / textures / scenes when they are destroyed and recreated with a new seed or reconfigured with a new seed.

## Configuration Options

There are a few configuration options and ways to use the `CachedResetWrapper`. One way is to modify how the reset states are generated. Below is the configuration dataclass that you can use and/or override when creating the wrapper

```python
@dataclass
class CachedResetsConfig:
    num_resets: Optional[int] = None
    """The number of reset states to cache. If none it will cache `num_envs` number of reset states."""
    device: Optional[Device] = None
    """The device to cache the reset states on. If none it will use the base environment's device."""
    seed: Optional[int] = None
    """The seed to use for generating the cached reset states."""

    def dict(self):
        return {k: v for k, v in asdict(self).items()}
```

For example to change the number of cached resets and the generation seed you can pass a dict as so

```python
env = CachedResetWrapper(env, config=dict(num_resets=16384, seed=0))
```

You can also manually pass in your own reset states and optionally observations paired with each reset.

```python
# env_states should be the result of env.get_state_dict(). It should be a dictionary where each leaf has the same batch size
# obs can be the observations you previously generated. It can also be none
env = CachedResetWrapper(env, reset_to_env_states=dict(env_states=env_states, obs=obs))
```

It may be useful to use the `tree` utility in ManiSkill if you want to e.g. concatenate multiple env_states values together from multiple calls to `env.get_state_dict` as so

```python
from mani_skill.utils import tree
state_dict_1 = env.get_state_dict()
# do something to the env
state_dict_2 = env.get_state_dict()
env_states = tree.cat([state_dict_1, state_dict_2])
env = CachedResetWrapper(env, reset_to_env_states=dict(env_states=env_states, obs=None))
```


## Performance

The following code snippet can quickly check the speed gains when using cached resets. For the example below with 256 envs, state observation mode
cached resets took on average about 0.004s while normal resets took 0.007s on a RTX 3080. With the rgb observation mode the difference is more staggering, with cached resets taking on average about 0.005s while normal resets took 0.167s.

```python
from mani_skill.utils.wrappers import CachedResetWrapper
import gymnasium as gym
import time

num_envs = 256
obs_mode = "rgb"
env = gym.make("StackCube-v1", obs_mode=obs_mode, num_envs=num_envs)
env = CachedResetWrapper(env)

trials = 100
start_time = time.time()
for i in range(trials):
    env.reset()
end_time = time.time()
print(f"Average time per cached reset: {(end_time - start_time) / trials} seconds")

env = gym.make("StackCube-v1", obs_mode=obs_mode, num_envs=num_envs)
# env = CachedResetWrapper(env)

trials = 100
start_time = time.time()
for i in range(trials):
    env.reset()
end_time = time.time()
print(f"Average time per reset: {(end_time - start_time) / trials} seconds")
```