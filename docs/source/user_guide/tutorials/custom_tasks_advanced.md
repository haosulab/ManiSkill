# Custom Tasks (Advanced Features)

This page covers nearly every feature useful for task building in ManiSkill.

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

Net contact forces are nearly the same as the pair-wise contact forces in terms of SAPIEN API but ManiSkill provides a convenient way to fetch this data for Actors and Articulations as so

```python
actor.get_net_contact_forces() # shape (N, 3)
articulation.get_net_contact_forces(link_names) # shape (N, len(link_names), 3)
```

## Scene Masks