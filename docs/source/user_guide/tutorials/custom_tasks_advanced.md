# Custom Tasks (Advanced Features)

This page covers nearly every feature useful for task building in ManiSkill. If you haven't already it is recommended to get a better understanding of how GPU simulation generally works described on [this page](../concepts/gpu_simulation.md). It can provide some good context for various terminology and ideas presented in this advanced features tutorial.

## Custom/Extra State

Reproducibility is generally important in task building. A common example is when trying to replay a trajectory that someone else generated, if that trajectory file is missing important state variables necessary for reconstructing the exact same initial task state, the trajectory likely would not replay correctly.

By default, `env.get_state_dict()` returns a state dictionary containing the entirety of simulation state, which consists of the poses and velocities of each actor and additionally qpos/qvel values of articulations.

In your own task you can define additional state data such as a eg `height` for a task like LiftCube which indicates how high the cube must be lifted for success. This would be your own variable and not included in `env.get_state_dict()` so to include it you can add the following two functions to your task class

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

ManiSkill defaults to actors/articulations when built to be built in every parallel sub-scene in the physx scene. This is not necessary behavior and you can control this by setting `scene_idxs`, which dictate which sub-scenes get the actor/articulation loaded into it. A good example of this done is in the PickSingleYCB task which loads a different geometry/object entirely in each sub-scene. This is done by effectively not creating one actor to pick up across all sub-scenes as you might do in PickCube, but a different actor per scene (which will be merged into one actor later).

```python
def _load_scene(self, options: dict):
    # ...
    for i, model_id in enumerate(model_ids):
        builder, obj_height = build_actor_ycb(
            model_id, self._scene, name=model_id, return_builder=True
        )
        builder.set_scene_idxs([i]) # spawn only in sub-scene i
        actors.append(builder.build(name=f"{model_id}-{i}"))
```
Here we have a list of YCB object ids in `model_ids`. For the ith `model_id` we create the ActorBuilder `builder` and set a scene mask so that only the ith sub-scene is True, the rest are False. Now when we call `builder.build` only the ith sub-scene has this particular object.

## Merging

### Merging Actors

In the [scene masks](#scene-masks) section we saw how we can restrict actors being built to specific scenes. However now we have a list of Actor objects and fetching the pose of each actor would need a for loop. The solution here is to create a new Actor that represents/views that entire list of actors via `Actor.merge` as done below (taken from the PickSingleYCB code). Once done, writing evaluation and reward functions become much easier as you can fetch the pose and other data of all the different actors with one batched attribute.



```python
from mani_skill.utils.structs import Pose
def _load_scene(self, options: dict):
    # ... code to create list of actors as shown in last code snippet
    obj = Actor.merge(actors, name="ycb_object")
    obj.pose.p # shape (N, 3)
    obj.pose.q # shape (N, 4)
    # etc.
```

Properties that exist regardless of geometry like object pose can be easily fetched after merging actors. This enables simple heterogenous simulation of diverse objects/geometries.

### Merging Articulations

WIP

## Task Sim Configurations

ManiSkill provides some reasonable default sim configuration settings but tasks with more complexity such as more objects, more possible collisions etc. may need more fine-grained control over various configurations, especially around GPU memory configuration.

In the drop down below is a copy of all the configurations possible

:::{dropdown} All sim configs
:icon: code

```
@dataclass
class GPUMemoryConfig:
    """A gpu memory configuration dataclass that neatly holds all parameters that configure physx GPU memory for simulation"""

    temp_buffer_capacity: int = 2**24
    """Increase this if you get 'PxgPinnedHostLinearMemoryAllocator: overflowing initial allocation size, increase capacity to at least %.' """
    max_rigid_contact_count: int = 2**19
    max_rigid_patch_count: int = (
        2**18
    )  # 81920 is SAPIEN default but most tasks work with 2**18
    heap_capacity: int = 2**26
    found_lost_pairs_capacity: int = (
        2**25
    )  # 262144 is SAPIEN default but most tasks work with 2**25
    found_lost_aggregate_pairs_capacity: int = 2**10
    total_aggregate_pairs_capacity: int = 2**10

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass
class SceneConfig:
    gravity: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))
    bounce_threshold: float = 2.0
    sleep_threshold: float = 0.005
    contact_offset: float = 0.02
    solver_iterations: int = 15
    solver_velocity_iterations: int = 1
    enable_pcm: bool = True
    enable_tgs: bool = True
    enable_ccd: bool = False
    enable_enhanced_determinism: bool = False
    enable_friction_every_iteration: bool = True
    cpu_workers: int = 0

    def dict(self):
        return {k: v for k, v in asdict(self).items()}

    # cpu_workers=min(os.cpu_count(), 4) # NOTE (stao): use this if we use step_start and step_finish to enable CPU workloads between physx steps.
    # NOTE (fxiang): PCM is enabled for GPU sim regardless.
    # NOTE (fxiang): smaller contact_offset is faster as less contacts are considered, but some contacts may be missed if distance changes too fast
    # NOTE (fxiang): solver iterations 15 is recommended to balance speed and accuracy. If stable grasps are necessary >= 20 is preferred.
    # NOTE (fxiang): can try using more cpu_workers as it may also make it faster if there are a lot of collisions, collision filtering is on CPU
    # NOTE (fxiang): enable_enhanced_determinism is for CPU probably. If there are 10 far apart sub scenes, this being True makes it so they do not impact each other at all


@dataclass
class DefaultMaterialsConfig:
    # note these frictions are same as unity
    static_friction: float = 0.3
    dynamic_friction: float = 0.3
    restitution: float = 0

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass
class SimConfig:
    spacing: int = 5
    """Controls the spacing between parallel environments when simulating on GPU in meters. Increase this value
    if you expect objects in one parallel environment to impact objects within this spacing distance"""
    sim_freq: int = 100
    """simulation frequency (Hz)"""
    control_freq: int = 20
    """control frequency (Hz). Every control step (e.g. env.step) contains sim_freq / control_freq physx simulation steps"""
    gpu_memory_cfg: GPUMemoryConfig = field(default_factory=GPUMemoryConfig)
    scene_cfg: SceneConfig = field(default_factory=SceneConfig)
    default_materials_cfg: DefaultMaterialsConfig = field(
        default_factory=DefaultMaterialsConfig
    )

    def dict(self):
        return {k: v for k, v in asdict(self).items()}
```
:::

To define a different set of default sim configurations, you can define a `_default_sim_cfg` property in your task class with the SimConfig etc. dataclasses as so

```python
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
class MyCustomTask(BaseEnv)
    # ...
    @property
    def _default_sim_cfg(self):
        return SimConfig(
            gpu_memory_cfg=GPUMemoryConfig(
                max_rigid_contact_count=self.num_envs * max(1024, self.num_envs) * 8,
                max_rigid_patch_count=self.num_envs * max(1024, self.num_envs) * 2,
                found_lost_pairs_capacity=2**26,
            )
        )
```

ManiSkill will fetch `_default_sim_cfg` after `self.num_envs` is set so you can also dynamically change configurations at runtime depending on the number of environments like it was done above. You usually need to change the default configurations when you try to run more parallel environments but SAPIEN will print critical errors about needing to increase one of the GPU memory configuration options.

Some of the other important configuration options and their defaults that are part of SimConfig are `spacing=5`, `sim_freq=100`, `control_freq=20`, and `'solver_iterations=15`. The physx timestep of the simulation is computed as `1 / sim_freq`, and the `control_freq` says that every `sim_freq/control_freq` physx steps we apply the environment action once and then fetch observation data to return to the user. 

- `spacing` is often a source of potential bugs since all sub-scenes live in the same physx scene and if objects in one sub-scene get moved too far they can hit another sub-scene if the `spacing` is too low
- higher `sim_freq` means more accurate simulation but slower physx steps
- higher `sim_freq/control_freq` ratio can often mean faster `env.step()` times
- higher `solver_iterations` increases simulation accuracy at the cost of speed. Notably environments like those with quadrupeds tend to set this value to 4 as they are much easier to simulate accurately without incurring significant sim2real issues.


Note the default `sim_freq, control_freq` values are tuned for GPU simulation and are generally usable (you shouldn't notice too many strange artifacts like objects sliding across flat surfaces).


<!-- TODO explain every option? -->

<!-- ## Defining Supported Robots and Robot Typing
 -->

## Mounted/Dynamically Moving Cameras

The custom tasks tutorial demonstrated adding fixed cameras to the PushCube task. ManiSkill+SAPIEN also supports mounting cameras to Actors and Links, which can be useful to e.g. have a camera follow a object as it moves around.

For example if you had a task with a baseketball in it and it's actor object is stored at `self.basketball`, in the `_sensor_configs` or `_human_render_camera_configs` properties you can do

```python
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
@property
def _sensor_configs(self)
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

:::{note}
Mounted cameras will generally be a little slower than static cameras unless you disable computing camera parameters. Camera parameters cannot be cached as e.g. the extrinsics constantly can change
:::

Mounted cameras also allow for some easy camera pose domain randomization [detailed further here](./domain_randomization.md#during-episode-initialization--resets). Cameras do not necessarily need to be mounted on standard objects, they can also be mounted onto "empty" actors that have no visual or collision shapes that you can create like so

```python
def _load_scene(self, options: dict):
    # ... your loading code
    self.cam_mount = self._scene.create_actor_builder().build_kinematic("camera_mount")
```

`self.cam_mount` has it's own pose data and if changed the camera will move with it.



## Before and After Control Step Hooks

You can run code before and after an action has been taken, both of which occur before observations are fetched. This can be useful for e.g. modifying some simulation states before observations are returned to the user. 

```python
def _before_control_step(self):
    # override this in your task class to run code before actions have been taken
    pass
def _after_control_step(self):
    # override this in your task class to run code after actions have been taken
```


## Modifying Simulation State outside of Reconfigure and Episode Initialization

In general it is not recommended to modify simulation state (e.g. setting an object pose) outside of the `_load_scene` function (called by reconfigure) or episode initialization in `_initialize_episode`. The reason is this can lead to sub-optimal task code that may make your task run slower than expected as in GPU simulation generally setting (and fetching) states takes some time. If you are only doing CPU simulation then this is generally fine and not slow at all.

Regardless there are some use cases to do so (e.g. change mounted camera pose every single timestep to a desired location). In such case, you must make sure you call `self._scene.gpu_apply_all()` after all of your state setting code runs in GPU simulation. This is to apply the changes you make to sim state and have it persist to the next environment time step.

Moreover, if you need to read up to data in GPU simulation, you should call `self._scene.gpu_fetch_all()` before reading any data like object pose. If you need up to date link pose data, you need to call `self._scene.px.gpu_update_articulation_kinematics()` before calling `self._scene.gpu_fetch_all()`.

:::{note} As we are constantly working to improve simulation speed and quality
it is possible the behavior of `self._scene.gpu_fetch_all()` may change in the future. If you want to call functions without worrying about 
changes you should use the original SAPIEN API for GPU data which is exposed via `self._scene.px` and gives more fine grained control about
what GPU data to fetch (which is more efficient than fetching all of it)
:::

## Plane Collisions

As all objects added to a sub-scene are also in the one physx scene containing all sub-scenes, plane collisions work differently since they extend to infinity. As a result, a plane collision spawned in two or more sub-scenes with the same poses will create a lot of collision issues and increase GPU memory requirements.

However as a user you don't need to worry about adding a plane collision in each sub-scene as ManiSkill automatically only adds one plane collision per given pose.