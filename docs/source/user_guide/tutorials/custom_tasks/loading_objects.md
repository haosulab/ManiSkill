# Loading Actors and Articulations

The [introductory tutorial](./intro.md) covered the overall process of building a custom task. This tutorial covers how to load a wider variety of objects, whether they are objects from asset datasets like [YCB](https://www.ycbbenchmarks.com/), or articulated datasets like [Partnet Mobility](https://sapien.ucsd.edu/browse). Loading these more complicated geometries enables you to build more complex and interesting robotics tasks.

## Loading Actors

ManiSkill provides two ways to load actors, loading directly from existing simulation-ready asset datasets, or via the lower-level ActorBuilder API.

### Loading from Existing Datasets

ManiSkill supports easily loading assets from existing datasets such as the YCB dataset. In the beta release this is the only asset database available, more will be provided once we finish integrating a 3D asset database system.

```python
from mani_skill.utils.building import actors
def _load_scene(self, options):
    builder = actors.get_actor_builder(
        self.scene,
        id=f"ycb:{model_id}",
    )
    builder.build(name="object")
```

### Using the ActorBuilder API

To build custom actors in python code, you first create the ActorBuilder as so in your task:

```python
def _load_scene(self, options):
    builder = self.scene.create_actor_builder()
```

Then you can use the standard SAPIEN API for creating actors, a tutorial of which can be found on the [SAPIEN actors tutorial documentation](https://sapien.ucsd.edu/docs/latest/tutorial/basic/create_actors.html)

## Loading Articulations

There are several ways to load articulations as detailed below.

### Loading from Existing Datasets

Like actors, ManiSkill supports easily loading articulated assets from existing datasets such as the Partnet Mobility dataset. In the beta release this is the only asset database available, more will be provided once we finish integrating a 3D asset database system.

```python
from mani_skill.utils.building import articulations
def _load_scene(self, options):
    builder = articulations.get_articulation_builder(
        self.scene, f"partnet-mobility:{model_id}"
    )
    builder.build(name="object")
```


### Using the ArticulationBuilder API

To build custom articulations in python code, you first create the ArticulationBuilder as so in your task:

```python
def _load_scene(self, options):
    builder = self.scene.create_articulation_builder()
```

Then you can use the standard SAPIEN API for creating articulations, a tutorial of which can be found on the [SAPIEN articulation tutorial documentation](https://sapien.ucsd.edu/docs/latest/tutorial/basic/create_articulations.html). You essentially just need to define what the links and joints are and how they connect. Links are created like Actors and can have visual and collision shapes added via the python API.

### Using the URDF Loader

If your articulation is defined with a URDF file, you can use a URDF loader to load that articulation and make modifications as needed.

```python
def _load_scene(self, options):
    loader = scene.create_urdf_loader()
    # the .parse function can also parse multiple articulations
    # actors and cameras but we only use the articulations
    articulation_builders, _, _ = loader.parse(str(urdf_path))
    builder = articulation_builders[0]
    builder.build(name="my_articulation")
```

You can also programmatically change various properties of articulations and their links prior to building it, see below for examples which range from fixing root links, collision mesh loading logic, and modifying physical properties. These can be useful for e.g. domain randomization

```python
def _load_scene(self, options):
    loader = scene.create_urdf_loader()
    
    # change friction values of all links
    loader.set_material(static_friction, dynamic_friction, restitution)
    # change friction values of specific links
    loader.set_link_material(link_name, static_friction, dynamic_friction, restitution)
    # change patch radius values of specific links
    loader.set_link_min_patch_radius(link_name, min_patch_radius)
    loader.set_link_patch_radius(link_name, patch_radius)
    # set density of all links
    loader.set_density(density)
    # set density of specific links
    loader.set_link_density(link_name, density)
    # fix/unfix root link in place
    loader.fix_root_link = True # or False
    # change the scale of the loaded articulation geometries (visual+collision)
    loader.scale = 1.0 # default is 1.0
    # if collision meshes contain multiple convex meshes
    # you can set this to True to try and load them
    loader.load_multiple_collisions_from_file = True

    articulation_builders, _, _ = loader.parse(str(urdf_path))
    builder = articulation_builders[0]
    builder.build(name="my_articulation")
```

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
