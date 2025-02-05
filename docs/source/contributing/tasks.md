# Tasks

ManiSkill is both an API as well as a source of high-quality simulated tasks in robotics. We encourage users to build off the ManiSkill API in their own codebases using our flexible API for CPU/GPU simulation. The documentation on how to use the ManiSkill API to do so is here: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks.html

For those who want to maintain a high standard for task building in addition to contributing official tasks to ManiSkill, we recommend the following
- Proper labelling of which robots/agents are supported
- Good use of typing when possible
- Documentation on what the task is, reset distribution/task randomizations, etc. 
- Video of what a solution in the task looks like
- (Optional): Dense rewards with working RL baseline

Each section below details the expectations for high quality, reproducible, and usable tasks, and optionally how to make a pull request to ManiSkill to add it.


## Typing and Labelling

A number of elements are necessary to make tasks findable, and filterable so users can determine whether they can use it for their various workflows.

### Registration

Tasks should be registered via

```python
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
@register_env("YourEnvID-v1", max_episode_steps=your_max_eps_steps)
class YourEnv(BaseEnv):
    # ...
```

with an appropriate name and the version. Generally the name should correspond with the file name for the code. Registration enables users to create the environment by string ID, as well as define a recommended number of max episode steps (which correlates with how long it takes to solve the task usually).

### Supported Agent/Robot Labelling

Each task always requires some sort of "agent" which could be a standard industrial robot arm or as simple as a cube that moves around. For a clearly well defined task you must define which agents are supported and work (namely does your code handle spawning that agent correctly and simulating it?). It is also strongly recommended to override the default typing of the agent variable defining a Union type over all robot classes that are supported.

```python
class YourEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
```

### Supported Reward Mode Labelling

Not all tasks permit easily definable/optimizable dense reward functions, nor is it required to write dense reward functions in ManiSkill given its difficulty. Regardless it is important to label which rewards the task supports. By default all tasks support `["sparse", "dense", "normalized_dense", "none"]`, but if you choose not write a dense reward function then you can do the following

```python
class YourEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["sparse", "none"]

    # note as dense is not labeled above, you do not need to override 
    # the compute_dense_reward or compute_normalized_dense_reward functions
```

## GPU Simulation Code/Testing

Whenever possible, task code should be written in batch mode (assuming all data in and out are batched by the number of parallel environments). This generally ensures that the task is then GPU simulatable, which is of great benefit to workflows that leverage sim data collection at scale.

GPU simulation also entails tuning the GPU simulation configurations. You can opt to do two ways, dynamic or fixed GPU simulation configurations.

A version of fixed configurations can be seen in `mani_skill/envs/tasks/push_cube.py` which defines the default

```python
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
class PushCube(BaseEnv):
    # ...
    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )
```

A version of dynamic configurations can be seen in `mani_skill/envs/tasks/dexterity/rotate_single_object_in_hand.py` which changes the configuration depending on the number of environments.

```python
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
class RotateSingleObjectInHand(BaseEnv):
    # ...
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=self.num_envs * max(1024, self.num_envs) * 8,
                max_rigid_patch_count=self.num_envs * max(1024, self.num_envs) * 2,
                found_lost_pairs_capacity=2**26,
            )
        )
```

For GPU simulation tuning, there are generally two considerations, memory and speed. It is recommended to set `gpu_memory_config` in such a way so that no errors are outputted when simulating as many as `4096` parallel environments with state observations on a single GPU. 

A simple way to test is to run the GPU sim benchmarking script on your already registered environment and check if any errors are reported

```bash
python -m mani_skill.examples.benchmarking.gpu_sim -e "YourEnv-v1" -n=4096 -o=state
```

Speed is also generally important and you can tune various sim configurations like solver iterations down up until the simulation is still stable. This part usually does not need to change but for certain tasks it may be required.

<!-- TODO
## Task Writing Semantics

While not strictly necessary, there are a few programming semantics/patterns to be aware of in order to make task code easier to write as well as being more readable. -->

## Task Card

Similar to how datasets and models can have associated cards describing them in detail, tasks also have "cards" that describe in sufficient detail for users to then use without having to dive into the code, wait for RL to work etc.

The task card must contain the following
- Tags/badges describing whether this task supports dense rewards and/or requires additional asset downloading
- Task description: Short few sentence description describing what the task is
- Supported robots: list of all robot uids supported (can copy from the code file)
- Randomizations: All randomizations performed during `_load_scene` and `_initialize_episode`, which can include e.g. geometry randomization or pose randomization of goals
- Success/Fail conditions: Details on what needs to occur for the task to succeed/fail. This is optional if a task does not have these conditions in the code
- Additional notes: Any additional comments that are not captured by the points above.
- Video of a successful demonstration of the task.

Examples of task cards are found throughout the [task documentation](../tasks/index.md)

## (Optional) Contributing the Task to ManiSkill Officially

When contributing to the task, make sure you do the following:

- The task code itself should have a reasonable unique name and be placed in `mani_skill/envs/tasks`.
- Added a demo video of the task being solved successfully (for each variation if there are several) to `figures/environment_demos`. The video should have ray-tracing on so it looks nicer! This can be done by replaying a trajectory with `human_render_camera_configs=dict(shader_pack="rt")` passed into `gym.make` when making the environment.
- Added a task card to `docs/source/tasks/index.md`.
