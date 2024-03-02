# Custom Environments/Tasks

Building custom tasks in ManiSkill is straightforward and flexible. ManiSkill provides a number of features to help abstract away most of the GPU memory management required for parallel simulation and rendering.

To build a custom environment/task in ManiSkill, it is comprised of the following core components

1. Robot(s) and Assets
2. Randomization
3. Success/Failure Condition
4. (Optional) Dense/shaped reward function
5. (Optional) Setting up cameras/sensors for observations and rendering/recording

## Adding Robot(s) and Assets

Loading these objects is done in the [`_load_actors`]() function.

## Randomization

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
