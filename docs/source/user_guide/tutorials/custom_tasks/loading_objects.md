# Loading Actors and Articulations

<!-- ```{include} ../../../../../mani_skill/envs/sapien_env.py
``` -->

## Reconfiguring and Optimization

In general loading is always quite slow, especially on the GPU so by default, ManiSkill reconfigures just once. Any call to `env.reset()` will not trigger a reconfiguration unless you call `env.reset(seed=seed, options=dict(reconfigure=True))` (seed is not needed but recommended if you are reconfiguring for reproducibility). 

However, during CPU simulation with just a single environment (or GPU simulation with very few environments) the loaded object geometries never get to change as reconfiguration doesn't happen more than once. This behavior can be changed by setting the `reconfiguration_freq` value of your task. 

The recommended way to do this is as follows (taken from the PickSingleYCB task):




```python
class PickSingleYCBEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Xmate3Robotiq, Fetch]
    goal_thresh = 0.025

    def __init__(
        self, *args, robot_uids="panda", robot_init_qpos_noise=0.02,
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        # ...
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )
```

A `reconfiguration_freq` value of 1 means every during every reset we reconfigure. A `reconfiguration_freq` of `k` means every `k` resets we reconfigure. A `reconfiguration_freq` of 0 (the default) means we never reconfigure again.

In general one use case of setting a positive `reconfiguration_freq` value is for when you want to simulate a task in parallel where each parallel environment is working with a different object/articulation and there are way more object variants than number of parallel environments. For machine learning / RL workflows, setting `reconfiguration_freq` to e.g. 10 ensures every 10 resets the objects being simulated on are randomized which can diversify the data collected for online training while keeping simulation fast by reconfiguring infrequently.
