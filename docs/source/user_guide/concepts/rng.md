# Reproducibility and Random Number Generation

Like many simulators, ManiSkill uses random number generators (RNG) for the randomization of anything in an environment, ranging from object geometry, object poses, textures etc.

However randomization introduces challenges for reproducibility when you generate a demonstration on your computer and want to replay it on another machine, especially when there is GPU simulation / parallelized environments where often times one seed decides the randomization done in each environment.

There are two ways to ensure reproducibility of trajectories/demonstrations, batch seeded RNG and environment state. Generally you only need to ensure reproducibility of random numbers during environment reconfiguration/loading (deciding which objects/textures to load into the environment) and episode initialization (deciding initial object positions, velocities etc).

## Reproducibility via RNG

To address this issue ManiSkill recommends using the `_batched_episode_rng` object of the environment. This batched RNG object is the same as [`np.random.RandomState`](https://numpy.org/doc/1.26/reference/random/legacy.html) but instead now all outputs have an additional batch dimension equal to the number of parallel environments. Moreover, the batched RNG object is seeded with a list of seeds corresponding to each parallel environment. Now regardless of the number of parallel environments the RNG generation is deterministic.


One downside of this approach is that RNG generation is slower than just batch generating random numbers directly with the default torch/numpy RNG functions. With that in mind we recommend at minimum using the batched episode rng for environment reconfiguration (which does not occur often during ML workflows during training). Episode initialization (which occurs once per episode) can be handled by the next approach of using environment states:

## Reproducibility via Env State

Environment state includes all the object's joint angles, joint velocities, poses, velocities. It does not include details such as object textures, poses of fixed cameras, robot controller stiffness etc.

Details such as object geometry/texture etc. are deterministic given the same seed and using the `_batched_episode_rng` object for randomization discussed above.

For other kinds of state that is included in environment state, to ensure reproducibility you can save these environment states and then set them after creating the environment. Assuming you used the same exact seed to ensure all the objects are the same, setting environment state will then ensure the objects are in the right places.

## Example Code/Scripts

TODO (stao): finish this section
