# Advanced Features

This page covers nearly every feature useful for task building in ManiSkill. If you haven't already it is recommended to get a better understanding of how GPU simulation generally works described on [this page](../../concepts/gpu_simulation.md). It can provide some good context for various terminology and ideas presented in this advanced features tutorial.

## Contact Forces on the GPU/CPU

### Pair-wise Contact Forces

You may notice that in some tasks like [PickCube-v1](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/pikc_cube.py) we call a function `self.agent.is_grasping(...)`. In ManiSkill, we leverage the pairwise impulses/forces API of SAPIEN to compute the forces between two objects. In the case of robots with two-finger grippers we check if both fingers are contacting a queried object. This is particularly useful for building better reward functions for faster RL. 

The API for querying pair-wise contact forces is unified between the GPU and CPU and is accessible via the `self.scene` object in your environment, accessible as so given a pair of actors/links of type `Actor | Link` to query via the ManiSkillScene object.

```python
self.scene: ManiSkillScene
forces = self.scene.get_pairwise_contact_forces(actor_1, link_2)
# forces is shape (N, 3) where N is the number of environments
```

:::{dropdown} Internal Implementation Caveats/Details
At the moment contacts work a bit different compared to CPU and GPU internally although they are unified under one interface/function for users.

On the GPU, one must create contact queries objects ahead of time and for performance reasons these are cached. Creating contact queries pauses all GPU computations similar to changing object poses/states. In ManiSkill this handled for users and it automatically creates queries if the queried contact forces are between objects that have not been queried before.
:::

For significantly more advanced/optimized usage of the contact forces API using the SAPIEN API directly instead of ManiSkill you can look at how [get_pairwise_contact_forces is implemented on GitHub](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/scene.py#L505-L539)

### Net Contact forces

Net contact forces are nearly the same as the pair-wise contact forces in terms of SAPIEN API but ManiSkill provides a convenient way to fetch this data for Actors and Articulations that works on CPU and GPU as so

```python
actor.get_net_contact_forces() # shape (N, 3), N is number of environments
articulation.get_net_contact_forces(link_names) # shape (N, len(link_names), 3)
```

## Scene Masks

ManiSkill defaults to actors/articulations when built to be built in every parallel sub-scene in the physx scene. This is not necessary behavior and you can control this by setting `scene_idxs`, which dictate which sub-scenes get the actor/articulation loaded into it. A good example of this feature is in the PickSingleYCB-v1 task which loads a different geometry/object entirely in each sub-scene. This is done by effectively not creating one actor to pick up across all sub-scenes as you might do in PickCube, but a different actor per scene (which will be merged into one actor later).

```python
class MyCustomTask(BaseEnv):
    # ...
    def _load_scene(self, options: dict):
        # sample a list of YCB object IDs for each parallel environment
        model_ids = self._batched_episode_rng.choice(self.all_model_ids)
        for i, model_id in enumerate(model_ids):
            builder, obj_height = build_actor_ycb(
                model_id, self.scene, name=model_id, return_builder=True
            )
            builder.set_scene_idxs([i]) # spawn only in sub-scene i
            actors.append(builder.build(name=f"{model_id}-{i}"))
```
Here we have a list of YCB object ids in `model_ids`. For the ith `model_id` we create the ActorBuilder `builder` and run `builder.set_scene_idxs([i])`. Now when we call `builder.build` only the ith sub-scene has this particular object.

Note that in this code we use `self._batched_episode_rng` to sample a list of model IDs for each parallel environment. This batched episode RNG object ensures the same models are sampled for the same list of seeds regardless of the number of parallel environments or if GPU/CPU simulation is being used. For more details on reproducibility and RNG see the page on [RNG](../../concepts/rng.md)

## Merging

ManiSkill is able to easily support task building with heterogeneous objects/articulations via a merging tool, which is way of viewing and reshaping existing data (on the GPU) into a single object to access from.

### Merging Actors

In the [scene masks](#scene-masks) section we saw how we can restrict actors being built to specific scenes. However now we have a list of Actor objects and fetching the pose of each actor would need a for loop. The solution here is to create a new Actor that represents/views that entire list of actors via `Actor.merge` as done below (taken from the PickSingleYCB-v1 code). Once done, writing evaluation and reward functions become much easier as you can fetch the pose and other data of all the different actors with one batched attribute.



```python
from mani_skill.utils.structs import Actor
class MyCustomTask(BaseEnv):
    # ...
    def _load_scene(self, options: dict):
        # ... code to create list of actors as shown in last code snippet
        obj = Actor.merge(actors, name="ycb_object")
        obj.pose.p # shape (N, 3)
        obj.pose.q # shape (N, 4)
        # etc.
```

Properties that exist regardless of geometry like object pose can be easily fetched after merging actors. This enables simple diverse simulation of diverse objects/geometries. Furthermore the `Actor.merge` function is fairly flexible, you do not necessarily need to ensure there the number of actors merged is equal to or even divisible by the number of parallel environments (stored in `self.num_envs`). Merge operations are simply a way to view batched data across different link objects via a single object.

### Merging Articulations

Similar to Actors, you can also merge Articulations. Note that this has a number of caveats since we allow merging articulations that may have completely different link/joint structures and DOFs.


```python
from mani_skill.utils.structs import Articulation
class MyCustomTask(BaseEnv):
    # ...
    def _load_scene(self, options: dict):
        # ... code to create list of articulations using scene masking
        art = Articulation.merge(articulations, name="name")
        art.pose.p # shape (N, 3)
        art.pose.q # shape (N, 4)
        art.qpos # shape (N, art.max_dof)
        art.qvel # shape (N, art.max_dof)
        art.qlimits # shape (N, art.max_dof, 2)
```

Properties that exist regardless of articulation include the base link's data (e.g. pose, velocities etc.). The qpos, qvel data of the merged articulation is also retrievable but note that it is padded now to the max dof, which you can get via `art.max_dof`. Retrieving the qpos values of joints you want can be a bit tricky, but that can be mostly handled by merging links which is detailed in the next section

### Merging Links

Similar to Actors, you can also merge any list of links sourced from any articulations you create. Upon merging, ManiSkill will also create a merged joint object that gives easy access to data of the the parent joints of each link without having to work with complex indexing of padded data. The joint merged object is only available if every link merged is not a root link as root links do not have a parent joint. Furthermore the `Link.merge` function is fairly flexible, you do not necessarily need to ensure there the number of links merged is equal to or even divisible by the number of parallel environments (stored in `self.num_envs`). Merge operations are simply a way to view batched data across different link objects via a single object.


```python
from mani_skill.utils.structs import Link
class MyCustomTask(BaseEnv):
    # ...
    def _load_scene(self, options: dict):
        # ... code to create list of articulations using scene masking
        # and then to select one link from each articulation
        link = Link.merge(links, name="my_link")
        link.pose.p # shape (N, 3)
        link.pose.q # shape (N, 4)
        link.joint # a merged joint object
        link.joint.qpos # shape (N, 1)
        link.joint.qvel # shape (N, 1)
```


## Custom/Extra State

Reproducibility is generally important in task building. A common example is when trying to replay a trajectory that someone else generated, if that trajectory file is missing important state variables necessary for reconstructing the exact same initial task state, the trajectory likely would not replay correctly.

By default, `env.get_state_dict()` returns a state dictionary containing the entirety of simulation state of actors and articulations in the state dict registry. Actor state is a flat `13` dimensional composed of 3D position, 4D quaternion, 3D linear velocity, and 3D angular velocity. Articulation state is a flat `13 + 2 * DOF` dimensional vector, with 13 dimensions corresponding to the root link's pose and velocities like actors, and the next `DOF` dimensions corresponding to the active joint positions and the last `DOF` dimensions corresponding to the active joint velocities.

By default, anytime an actor or articulation is built, it is automatically added to the state dict registry which can be viewed via `env.scene.state_dict_registry`.

### Handling Custom States

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

### Handling Heterogeneous Simulation States

Many tasks like OpenCabinetDrawer-v1 and PegInsertionSide-v1 support heterogeneous simulation where each environment has a different object/geometry/dof. For these tasks often times during scene loading you run a for loop to create each different object in each parallel environment. However in doing so you have to give them each a different name which causes inconsistency in the environment state as now the number of environments changes the shape of the state dictionary. To handle this you need to remove the per-environment actor/articulation from the state dictionary registry and add in the merged version. See sections on how to use [scene masks](#scene-masks) and [merging](#merging) for information on how to build heterogeneous simulated environments.

```python
class MyCustomTask(BaseEnv):
    # ...
    def _load_scene(options):
        # ... 
        for i in range(self.num_envs):
            # build your object ...
            name = f"object_{i}"
            object_i = builder.build(name=name)
            self.remove_from_state_dict_registry(object_i)

        self.object = Actor.merge(objects, name="object") 
        # or if its articulation you use Articulation.merge
        self.add_to_state_dict_registry(self.object)
        # note that some tasks merge links, but links do not need to be registered to the state dict registry
        # link state is included with merged articulations
```


## Getting Collision Shapes + Bounding Boxes

Both Actor and Articulation objects have a `get_first_collision_mesh` API which returns a `trimesh.Trimesh` object in the world frame representing the combined collision mesh from the trimesh package. Currently we do not have a batched mesh processing library and rely on trimesh still, so we currently only support getting collision mesh of an actor/articulation when all the managed objects are the same.

Once a collision mesh is obtained, you can use it to get a bounding box. Note that you cannot get this collision mesh until after the `_load_scene` function. This is because for GPU simulation, we currently cannot get the correct collision mesh prior to the GPU buffers being initialized (which occurs at the end of reconfiguration).

A use case for bounding boxes is to spawn objects so that they are upright and don't intersect `z=0`. In the PickSingleYCB task we do the following.

```python
def _after_reconfigure(self, options: dict):
    self.object_zs = []
    # self._objs is a list of Actor objects
    for obj in self._objs:
        collision_mesh = obj.get_first_collision_mesh()
        # this value is used to set object pose so the bottom is at z=0
        self.object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
    self.object_zs = common.to_tensor(self.object_zs)
```

While in ManiSkill we do this to set object poses for tasks like PickSingleYCB and OpenCabinetDrawer, it is recommended to avoid this and to cache bounding box / collision mesh information in a file (e.g. JSON) and to load that when the environment is created.

## Task Sim Configurations

ManiSkill provides some reasonable default sim configuration settings but tasks with more complexity such as more objects, more possible collisions etc. may need more fine-grained control over various configurations, especially around GPU memory configuration.

In the drop down below is a copy of all the configurations possible

:::{dropdown} All sim configs
:icon: code

```python
@dataclass
class GPUMemoryConfig:
    """A gpu memory configuration dataclass that neatly holds all parameters that configure physx GPU memory for simulation"""

    temp_buffer_capacity: int = 2**24
    """Increase this if you get 'PxgPinnedHostLinearMemoryAllocator: overflowing initial allocation size, increase capacity to at least %.' """
    max_rigid_contact_count: int = 2**19
    """Increase this if you get 'Contact buffer overflow detected'"""
    max_rigid_patch_count: int = (
        2**18
    )  # 81920 is SAPIEN default but most tasks work with 2**18
    """Increase this if you get 'Patch buffer overflow detected'"""
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
    rest_offset: float = 0
    solver_position_iterations: int = 15
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
    spacing: float = 5
    """Controls the spacing between parallel environments when simulating on GPU in meters. Increase this value
    if you expect objects in one parallel environment to impact objects within this spacing distance"""
    sim_freq: int = 100
    """simulation frequency (Hz)"""
    control_freq: int = 20
    """control frequency (Hz). Every control step (e.g. env.step) contains sim_freq / control_freq physx simulation steps"""
    gpu_memory_config: GPUMemoryConfig = field(default_factory=GPUMemoryConfig)
    scene_config: SceneConfig = field(default_factory=SceneConfig)
    default_materials_config: DefaultMaterialsConfig = field(
        default_factory=DefaultMaterialsConfig
    )

    def dict(self):
        return {k: v for k, v in asdict(self).items()}
```
:::

To define a different set of default sim configurations, you can define a `_default_sim_config` property in your task class with the SimConfig etc. dataclasses as so

```python
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
class MyCustomTask(BaseEnv):
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

ManiSkill will fetch `_default_sim_config` after `self.num_envs` is set so you can also dynamically change configurations at runtime depending on the number of environments like it was done above. You usually need to change the default configurations when you try to run more parallel environments, and SAPIEN will print critical errors about needing to increase one of the GPU memory configuration options.

Some of the other important configuration options and their defaults that are part of SimConfig are `spacing=5`, `sim_freq=100`, `control_freq=20`, and `'solver_iterations=15`. The physx timestep of the simulation is computed as `1 / sim_freq`, and the `control_freq` says that every `sim_freq/control_freq` physx steps we apply the environment action once and then fetch observation data to return to the user. 

- `spacing` is often a source of potential bugs since all sub-scenes live in the same physx scene and if objects in one sub-scene get moved too far they can hit another sub-scene if the `spacing` is too low
- higher `sim_freq` means more accurate simulation but slower physx steps and thus slower `env.step()` times.
- lower `sim_freq/control_freq` ratio can often mean faster `env.step()` times although some online algorithms like RL may not learn faster. The default is 100/20 = 5
- higher `solver_iterations` increases simulation accuracy at the cost of speed. Notably environments like those with quadrupeds tend to set this value to 4 as they are much easier to simulate accurately without incurring significant sim2real issues.


Note the default `sim_freq, control_freq` values are tuned for GPU simulation and are generally usable (you shouldn't notice too many strange artifacts like objects sliding across flat surfaces).


<!-- TODO explain every option? -->

<!-- ## Defining Supported Robots and Robot Typing
 -->

## Mounted/Dynamically Moving Cameras

The custom tasks tutorial demonstrated adding fixed cameras to the PushCube task. ManiSkill+SAPIEN also supports mounting cameras to Actors and Links, which can be useful to e.g. have a camera follow a object as it moves around.

For example if you had a task with a basketball in it and its actor object is stored at `self.basketball`, in the `_default_sensor_configs` or `_default_human_render_camera_configs` properties you can do

```python
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
class MyCustomTask(BaseEnv):
    # ...
    @property
    def _default_sensor_configs(self)
        # look towards the center of the basketball from a positon that is offset
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

Mounted cameras also allow for some easy camera pose domain randomization [detailed further here](../domain_randomization.md#during-episode-initialization--resets). Cameras do not necessarily need to be mounted on standard objects, they can also be mounted onto "empty" actors that have no visual or collision shapes that you can create like so

```python
class MyCustomTask(BaseEnv):
    # ...
    def _load_scene(self, options: dict):
        # ... your loading code
        self.cam_mount = self.scene.create_actor_builder().build_kinematic("camera_mount")
```

`self.cam_mount` has its own pose data and if changed the camera will move with it.



## Before and After Control Step Hooks

You can run code before and after an action has been taken, both of which occur before observations are fetched. This can be useful for e.g. modifying some simulation states before observations are returned to the user. 

```python
class MyCustomTask(BaseEnv):
    # ...
    def _before_control_step(self):
        # override this in your task class to run code before actions have been taken
        pass
    def _after_control_step(self):
        # override this in your task class to run code after actions have been taken
        pass
```


## Modifying Simulation State outside of Reconfigure and Episode Initialization

In general it is not recommended to modify simulation state (e.g. setting an object pose) outside of the `_load_scene` function (called by reconfigure) or episode initialization in `_initialize_episode`. The reason is this can lead to sub-optimal task code that may make your task run slower than expected as in GPU simulation generally setting (and fetching) states takes some time. If you are only doing CPU simulation then this is generally fine and not slow at all.

Regardless there are some use cases to do so (e.g. change mounted camera pose every single timestep to a desired location). In such cases, you must make sure you call `self.scene.gpu_apply_all()` after all of your state setting code runs during GPU simulation. This applies the changes you make to sim state and ensures it persists to the next environment time step.

Moreover, if you need access to up to date data in GPU simulation, you should call `self.scene.gpu_fetch_all()` before reading any data like object pose. If you need up to date link pose data, you need to call `self.scene.px.gpu_update_articulation_kinematics()` before calling `self.scene.gpu_fetch_all()`.

:::{note} As we are constantly working to improve simulation speed and quality
it is possible the behavior of `self.scene.gpu_fetch_all()` may change in the future. If you want to call functions without worrying about 
changes you should use the original SAPIEN API for GPU data which is exposed via `self.scene.px` and gives more fine grained control about
what GPU data to fetch (which is more efficient than fetching all of it)
:::

## Plane Collisions

As all objects added to a sub-scene are also in the one physx scene containing all sub-scenes, plane collisions work differently since they extend to infinity. As a result, a plane collision spawned in two or more sub-scenes with the same poses will create a lot of collision issues and increase GPU memory requirements.

However as a user you don't need to worry about adding a plane collision in each sub-scene as ManiSkill automatically only adds one plane collision per given pose.

## Dynamically Updating Simulation Properties

See the page on [domain randomization](../domain_randomization.md) for more information on modifying various simulation (visual/physical) properties on the fly.