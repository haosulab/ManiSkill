# Custom Tasks

Building custom tasks in ManiSkill is straightforward and flexible. ManiSkill provides a number of features to help abstract away most of the GPU memory management required for parallel simulation and rendering.

To build a custom task in ManiSkill, it is comprised of the following core components

1. Loading (Robots, Assets, Sensors, etc.)
2. Initialization / Randomization
3. Success/Failure Condition
4. (Optional) Dense/shaped reward function
5. (Optional) Setting up cameras/sensors for observations and rendering/recording

This tutorial will first cover each of the core components, and then showcase 3 different tutorial tasks ([PushCube](#example-task-1-push-cube), [PickSingleYCB](#example-task-2-pick-single-ycb), [OpenCabinetDrawer](#example-task-3-open-cabinet-drawer)) that showcase how to use most of the features in ManiSkill.

## Loading

At the start of any task, you must load in all objects (robots, assets, articulations, lighting etc.) into the scene. Loading these objects is done in the `_load_actors` function of your custom task class.

To build objects 

### Reconfiguring and Optimization

In general loading is always quite slow, especially on the GPU so by default, ManiSkill reconfigures just once, any call to `env.reset()` will not trigger a reconfiguration unless you call `env.reset(seed=seed, options=dict(reconfigure=True))` (seed is not needed but recommended if you are reconfiguring for reproducibility).

If you want calls to `env.reset()` to by default reconfigure, you can set a default value for `reconfigure_freq` in your task's `__init__` function

```python
def __init__(self, *args, robot_uids="panda", reconfigure_freq=1, **kwargs):
    super().__init__(*args, robot_uids=robot_uids, reconfiguration_freq=reconfiguration_freq, **kwargs)
```

A `reconfiguration_freq` value of 1 means every during every reset we reconfigure. A `reconfiguration_freq` of `k` means every `k` resets we reconfigure. A `reconfiguration_freq` of 0 (the default) means we never reconfigure again.

In general the 


## Initialization / Randomization

Task initialization and randomization is handled in the [`_initalize_actors`]() function.

## Success/Failure Conditions

For each task, we need to determine if it has been completed successfully.


## Advanced - Diverse objects/articulations

TODO (stao)
IDEAL API?
V1

call build_actor/build_articulation for each unique one and set mask and then build it.
then call `merge_actors(actors: List[Actor]) -> Actor`... and it will just merge all actors? (a bit easier?)

V2

build entity yourself in each sub scene, then merge them all with Actor.create_from_entities(...) or something

### Articulations

```python
def Articulation.merge_articulations(articulations: List[Articulation]) -> Articulation:
  ...
```

As articulations can all have different DOFs, different links entirely, not all properties of articulations can be used easily, need masking

Shared: 
- root poses
- bounding box

Not shared
- link children and parents


## Example Task 1: Push Cube

## Example Task 2: Pick Single YCB

The goal of this example task is to demonstrate how to make task building with heterogenous object geometries easy via the actor merging API. Building tasks with heteroenous objects allows for easier diverse data collection and generaliable policy training. The complete task code is at [mani_skill2/envs/tasks/pick_single_ycb.py](https://github.com/haosulab/ManiSkill2/tree/main/mani_skill2/envs/tasks/pick_single_ycb.py)

Previously in PushCube, we showed how one can simply create a single object like a cube, and ManiSkill will automatically spawn that cube in every sub-scene. To create a different object in each sub-scene, in this case a random object sampled from the YCB object Dataset, you must do this part yourself. As a user you simply write code to decide which sub-scene will have which object. This is done by creating an actor builder as usual, but now setting a scene mask to decide which sub-scenes have this object and which do not.

```python
for i, model_id in enumerate(model_ids):
  builder, obj_height = build_actor_ycb(
      model_id, self._scene, name=model_id, return_builder=True
  )
  scene_mask = np.zeros(self.num_envs, dtype=bool)
  scene_mask[i] = True
  builder.set_scene_mask(scene_mask)
  actors.append(builder.build(name=f"{model_id}-{i}"))
  self.obj_heights.append(obj_height)
```

The snippet above will now create a list of `Actor` objects, but this makes fetching data about these different actors complicated because you would have to loop over each one. Here you can now use the merge API shown below to simply merge all of these `Actor` objects in the `actors` list into one object that you can then fetch data shared across all objects like pose, linear velocity etc.

```python
self.obj = Actor.merge(actors, name="ycb_object")
```


## Example Task 3: Open Cabinet Drawer