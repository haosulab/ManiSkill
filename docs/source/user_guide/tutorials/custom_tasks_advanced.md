# Custom Tasks (Advanced Features)

This page covers nearly every feature useful for task building in ManiSkill. If you haven't already it is recommended to get a better understanding of how GPU simulation generally works described on [this page](../concepts/gpu_simulation.md). It can provide some good context for various terminology and ideas presented in this advanced features tutorial.

## Custom/Extra State

Reproducibility is generally important in task building. A common example is when trying to replay a trajectory that someone else generated, if that trajectory file is missing important state variables necessary for reconstructing the exact same initial task state, the trajectory likely would not replay correctly.

By default, `env.get_state_dict()` returns a state dictionary containing the entirety of simulation state, which consists of the poses and velocities of each actor and additionally qpos/qvel values of articulations.

In your own task you can define additional state data such as a eg `height` for a task like LiftCube which indicates how high the cube must be lifted for success. This would your own variable and not included in `env.get_state_dict()` so to include it you can add the following two functions to your task class

```python
class MyCustomTask(BaseEnv):
    # ...
    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict["height"] = self.height
    def set_state_dict(self, state_dict):
        super().set_state_dict(state_dict)
        self.height = state_dict["height"]
```

Now recorded trajectories of your task will include the height as part of the environment state and so you can replay the trajectory with just environment states perfectly in the sense that the robot path is the same and the evaluation/reward metrics output the same at each time step.

## Contact Forces on the GPU/CPU

### Pair-wise Contact Forces

You may notice that in some tasks like [PickCube-v1](https://github.com/haosulab/ManiSkill2/blob/dev/mani_skill/envs/tasks/pikc_cube.py) we call a function `self.agent.is_grasping(self.)`. In ManiSkill, we leverage the pairwise impulses/forces API of SAPIEN to compute the forces betewen two objects. In the case of robots with two-finger grippers we check if both fingers are contacting a queried object. This is particularly useful for building better reward functions.


#### On the CPU
If you have an two `Actor` or `Link` class objects, call them `x1, x2`, you can generate the contact force between them via

```python
from mani_skill.utils import sapien_utils
impulses = sapien_utils.get_pairwise_contact_impulse(
    scene.get_contacts(), x1._bodies[0].entity, x2._bodies[0].entity
)
```

#### On the GPU

To get pair-wise contact forces on the GPU you have to first generate a list of 2-tuples of bodies you wish to compute contacts between. If you have an two `Actor` or `Link` class objects, call them `x1, x2`, you can generate the contact force between them via


```python
# this generates two N 2-tuples where N is the number of parallel envs
body_pairs = list(zip(x1._bodies, x2._bodies))
```
You are not restricted to having to use all of the `_bodies`, you can pick any subset of them if you wish.

Then you can generate and cache the pair impulse query as so
```python
query = scene.px.gpu_create_contact_pair_impulse_query(body_pairs)
```

To fetch the impulses of the contacts between bodies
```python
# impulses divided by the physx timestep gives forces in newtons
contacts = query.cuda_impulses.torch().clone() / scene.timestep
# results in a array of shape (len(body_pairs), 3)
```

There are plans to make a simpler managed version of the SAPIEN pair-wise contacts API. If you want to use it yourself you can check out how it is used to check two-finger grasps on the [Panda Robot](https://github.com/haosulab/ManiSkill2/blob/dev/mani_skill/agents/robots/panda/panda.py) or generate tactile sensing data on the [Allegro Hand](https://github.com/haosulab/ManiSkill2/blob/dev/mani_skill/agents/robots/allegro_hand/allegro_touch.py).

### Net Contact forces

Net contact forces are nearly the same as the pair-wise contact forces in terms of SAPIEN API but ManiSkill provides a convenient way to fetch this data for Actors and Articulations that works on CPU and GPU as so

```python
actor.get_net_contact_forces() # shape (N, 3)
articulation.get_net_contact_forces(link_names) # shape (N, len(link_names), 3)
```

## Scene Masks

ManiSkill defaults to actors/articulations when built to be built in every parallel sub-scene in the physx scene. This is not necessary behavior and you can control this via scene masks, which dictate where sub-scenes get the actor/articulation loaded into it and which do not. A good example of this done is in the PickSingleYCB task which loads a different geometry/object entirely in each sub-scene. This is done by effectively not creating one actor to pick up across all sub-scenes as you might do in PickCube, but a different actor per scene (which will be merged into one actor later).

```python
for i, model_id in enumerate(model_ids):
    builder, obj_height = build_actor_ycb(
        model_id, self._scene, name=model_id, return_builder=True
    )
    scene_mask = np.zeros(self.num_envs, dtype=bool)
    scene_mask[i] = True
    builder.set_scene_mask(scene_mask)
    actors.append(builder.build(name=f"{model_id}-{i}"))
```
Here we have a list of YCB object ids in `model_ids`. For the ith `model_id` we create the ActorBuilder `builder` and set a scene mask so that only the ith sub-scene is True, the rest are False. Now when we call `builder.build` only the ith sub-scene has this particular object.

## Merging

### Merging Actors

In the [scene masks](#scene-masks) section we saw how we can restrict actors being built to specific scenes. However now we have a list of Actor objects and fetching the pose of each actor would need a for loop. The easy solution here is to create a new Actor that represents/views that entire list of actors via `Actor.merge` as done below (taken from the PickSingleYCB code).

```python
obj = Actor.merge(actors, name="ycb_object")
```

Now we have the following useful behaviors which can make writing evaluation and reward functions a breeze

```python
obj.pose.p # shape (N, 3)
obj.pose.q # shape (N, 4)
# etc.
```
effectively properties that exist regardless of geometry like object pose can be easily fetched after merging actors.

### Merging Articulations

WIP

## Mounted/Dynamically Moving Cameras

The custom tasks tutorial demonstrated adding fixed cameras to the PushCube task. ManiSkill+SAPIEN also supports mounting cameras to Actors and Links, which can be useful to e.g. have a camera follow a object as it moves around.

For example if you had a task with a baseketball in it and it's actor object is stored at `self.basketball`, in the `_register_sensors` or `_register_human_render_cameras` functions you can do

```python

def _register_sensors(self)
    # look towards the center of the baskeball from a positon that is offset
    # (0.3, 0.3, 0.5) away from the basketball
    pose = sapien_utils.look_at(eye=[0.3, 0.3, 0.5], target=[0, 0, 0])
    return [
        CameraConfig(
            uid="ball_cam", p=pose.p, q=pose.q, width=128,
            height=128, fov=np.pi / 2, near=0.01,
            far=100, mount=self.basketball
        )]
```

<!-- TODO show video of example -->