# Loading Actors and Articulations

The [introductory tutorial](./intro.md) covered the overall process of building a custom task. This tutorial covers how to load a wider variety of objects, whether they are simple geometric objects, objects from asset datasets like [YCB](https://www.ycbbenchmarks.com/), or articulated objects from datasets like [PartNet Mobility](https://sapien.ucsd.edu/browse).

Actors and articulations are created during `_load_scene`. They are then repositioned, randomized, and reset during `_initialize_episode`. This split is important for GPU simulation: loading geometry is expensive and usually happens only when the scene is reconfigured, while setting poses and joint states is cheap and can happen every episode.

## Actors

An actor is a single rigid body. In a manipulation task, movable cubes, mugs, goal markers, tables, and camera mounts are all actors. Actors can be dynamic, kinematic, or static:

- **dynamic** actors are simulated by physics and can move after contacts or forces.
- **kinematic** actors can be moved by your code but are not pushed around by physics.
- **static** actors cannot move after loading. They are useful for floors, walls, and fixed scenery.

### Building Primitive Actors

Use `self.scene.create_actor_builder()` when you want to define an actor directly in Python. Add one or more collision shapes for physics, add one or more visual shapes for rendering, set an initial pose, and then build the actor.

```python
import numpy as np
import sapien


def _load_scene(self, options):
    builder = self.scene.create_actor_builder()

    # Collision shapes are used by the physics solver.
    builder.add_box_collision(half_size=[0.02, 0.02, 0.02])

    # Visual shapes are used by cameras and human rendering.
    builder.add_box_visual(
        half_size=[0.02, 0.02, 0.02],
        material=sapien.render.RenderMaterial(
            base_color=np.array([0.1, 0.2, 0.8, 1.0])
        ),
    )

    # Initial poses should avoid intersections with other loaded objects.
    builder.set_initial_pose(sapien.Pose(p=[0, 0, 0.02]))

    # Dynamic actors are affected by contacts, gravity, and forces.
    self.cube = builder.build_dynamic(name="cube")
```

`build_dynamic`, `build_kinematic`, and `build_static` are convenience wrappers around `build`. The lower-level pattern below is equivalent to `build_dynamic`:

```python
builder.set_physx_body_type("dynamic")
self.cube = builder.build(name="cube")
```

Built actor names must be unique within the scene. ManiSkill stores the returned object in `self.scene.actors[name]` and includes it in the environment state dictionary. Keep a reference such as `self.cube` when the task needs to query or reset that object later.

### Collision and Visual Shapes

Most task objects should have both collision and visual shapes. Collision-only objects affect physics but are invisible. Visual-only objects are useful for goal markers and overlays because they do not affect physics.

```python
import sapien


def _load_scene(self, options):
    marker_builder = self.scene.create_actor_builder()
    marker_builder.add_sphere_visual(
        radius=0.03,
        material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 1]),
    )
    marker_builder.set_initial_pose(sapien.Pose(p=[0.2, 0, 0.01]))

    # Kinematic visual marker: movable by task code, ignored by contacts.
    self.goal_marker = marker_builder.build_kinematic(name="goal_marker")
```

For common shapes, ManiSkill also provides helpers in `mani_skill.utils.building.actors`.

```python
import sapien
from mani_skill.utils.building import actors


def _load_scene(self, options):
    self.goal_region = actors.build_red_white_target(
        self.scene,
        radius=0.1,
        thickness=1e-5,
        name="goal_region",
        add_collision=False,
        body_type="kinematic",
        initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
    )
```

### Loading Actor Meshes

Use mesh-based shapes when primitive shapes are not enough. Visual meshes can be arbitrary renderable mesh files. Collision meshes should usually be convex or decomposed into multiple convex pieces for stable simulation.

```python
import sapien


def _load_scene(self, options):
    builder = self.scene.create_actor_builder()
    builder.add_visual_from_file("assets/object.glb")
    builder.add_multiple_convex_collisions_from_file("assets/object_collision.obj")
    builder.set_initial_pose(sapien.Pose(p=[0, 0, 0.1]))
    self.object = builder.build_dynamic(name="object")
```

Use non-convex collision meshes only for static scenery. Dynamic non-convex collision is slower and less stable than convex collision.

### Loading Actors from Existing Datasets

ManiSkill includes builders for asset datasets such as YCB. These dataset builders return an `ActorBuilder`, so you can still set the initial pose and build type before calling `build`.

```python
import sapien
from mani_skill.utils.building import actors


def _load_scene(self, options):
    builder = actors.get_actor_builder(
        self.scene,
        id=f"ycb:{model_id}",
    )
    builder.set_initial_pose(sapien.Pose(p=[0, 0, 0.1]))
    self.object = builder.build(name="object")
```

### Building Actors in Only Some Parallel Environments

By default a builder creates one managed actor for every parallel environment. Use `set_scene_idxs` to create an actor only in a subset of environments. This is useful for heterogeneous tasks where different environments load different assets.

```python
import sapien


def _load_scene(self, options):
    builder = self.scene.create_actor_builder()
    builder.add_box_collision(half_size=[0.02, 0.02, 0.02])
    builder.add_box_visual(half_size=[0.02, 0.02, 0.02])
    builder.set_scene_idxs([0, 2, 4])
    builder.set_initial_pose(sapien.Pose(p=[0, 0, 0.02]))
    partial_actor = builder.build_dynamic(name="partial_actor")
```

If you load different actors in different subsets of environments and want one object handle for later pose queries or resets, merge the returned actors.

```python
import sapien
from mani_skill.utils.building import actors
from mani_skill.utils.structs import Actor


def _load_scene(self, options):
    objects = []
    for scene_idx, model_id in enumerate(model_ids):
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
        builder.set_scene_idxs([scene_idx])
        builder.set_initial_pose(sapien.Pose(p=[0, 0, 0.1]))
        objects.append(builder.build(name=f"object_{scene_idx}"))

    self.object = Actor.merge(objects, name="object")
```

Merged actors are useful as a batched view for poses and state, but some geometry queries such as collision mesh fetching are not supported on merged actors because the managed objects may have different meshes.

### Resetting Actor Poses

Set randomized or episode-specific actor poses in `_initialize_episode`, not `_load_scene`. ManiSkill's `Pose` wrapper supports batched poses and works on both CPU and GPU simulation.

```python
import torch
from mani_skill.utils.structs import Pose


def _initialize_episode(self, env_idx: torch.Tensor, options):
    with torch.device(self.device):
        b = len(env_idx)
        xyz = torch.zeros((b, 3))
        xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
        xyz[:, 2] = 0.02
        self.cube.set_pose(Pose.create_from_pq(p=xyz, q=[1, 0, 0, 0]))
```

During GPU simulation, setters such as `set_pose`, `set_linear_velocity`, and `set_angular_velocity` are automatically masked to the environments being reset. You normally use `env_idx` only to choose the batch size and sample reset data.

## Articulations

An articulation is a tree of links connected by joints. Robot arms, grippers, cabinets, drawers, doors, and many PartNet Mobility objects are articulations. Build or load articulations in `_load_scene`, then set their root pose, joint positions, and joint velocities in `_initialize_episode`.

### Loading Articulations from Existing Datasets

ManiSkill supports articulated datasets such as PartNet Mobility.

```python
import sapien
from mani_skill.utils.building import articulations


def _load_scene(self, options):
    builder = articulations.get_articulation_builder(
        self.scene,
        f"partnet-mobility:{model_id}",
    )
    # Avoid initial intersections, especially in GPU simulation.
    builder.initial_pose = sapien.Pose(p=[0, 0, 0.5])
    self.object = builder.build(name="object")
```

### Building Articulations in Python

Use `self.scene.create_articulation_builder()` when you want to define links and joints directly. Each link gets collision and visual shapes just like an actor. Joint records define how each child link moves relative to its parent.

```python
import numpy as np
import sapien


def _load_scene(self, options):
    builder = self.scene.create_articulation_builder()

    root = builder.create_link_builder()
    root.set_name("base")
    root.add_box_collision(half_size=[0.05, 0.05, 0.02])
    root.add_box_visual(
        half_size=[0.05, 0.05, 0.02],
        material=sapien.render.RenderMaterial(base_color=[0.4, 0.4, 0.4, 1]),
    )

    link = builder.create_link_builder(parent=root)
    link.set_name("arm")
    link.add_capsule_collision(radius=0.01, half_length=0.12)
    link.add_capsule_visual(
        radius=0.01,
        half_length=0.12,
        material=sapien.render.RenderMaterial(base_color=[0.2, 0.6, 0.9, 1]),
    )
    link.set_joint_name("base_to_arm")
    link.set_joint_properties(
        type="revolute",
        limits=[[-np.pi / 2, np.pi / 2]],
        pose_in_parent=sapien.Pose(p=[0, 0, 0.02]),
        pose_in_child=sapien.Pose(p=[0, 0, -0.12]),
        friction=0.0,
        damping=0.1,
    )

    builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
    self.arm = builder.build(name="arm")
```

For most custom robots and articulated assets, a URDF or MJCF file is easier to maintain than building every link in Python. Direct Python articulation building is most useful for small mechanisms, procedural tasks, and tests.

### Resetting Articulation State

Use root-pose and joint-state setters in `_initialize_episode`. Joint arrays are batched across environments, with shape `(num_envs, dof)` for full resets.

```python
import torch
from mani_skill.utils.structs import Pose


def _initialize_episode(self, env_idx: torch.Tensor, options):
    with torch.device(self.device):
        b = len(env_idx)
        self.arm.set_root_pose(
            Pose.create_from_pq(
                p=torch.tensor([[0, 0, 0.1]]).repeat(b, 1),
                q=[1, 0, 0, 0],
            )
        )
        self.arm.set_qpos(torch.zeros((b, self.arm.max_dof)))
        self.arm.set_qvel(torch.zeros((b, self.arm.max_dof)))
```

You can inspect joint and link names from the returned articulation. This is useful when you need to set only a subset of joints or query a specific link.

```python
joint_names = [joint.name for joint in self.arm.joints]
link_names = [link.name for link in self.arm.links]
```

### Using the URDF Loader

If your articulation is defined with a URDF file, you can use a URDF loader to load that articulation and make modifications as needed.

```python
def _load_scene(self, options):
    loader = self.scene.create_urdf_loader()
    # the .parse function can also parse multiple articulations
    # actors and cameras but we only use the articulations
    articulation_builders = loader.parse(str(urdf_path))["articulation_builders"]
    builder = articulation_builders[0]
    # choose a reasonable initial pose that doesn't intersect other objects
    # this matters a lot for articulations in GPU sim or else simulation bugs can occur
    builder.initial_pose = sapien.Pose(p=[0, 0, 0.5])
    builder.build(name="my_articulation")
```

You can also programmatically change various properties of articulations and their links prior to building it, see below for examples which range from fixing root links, collision mesh loading logic, and modifying physical properties. These can be useful for e.g. domain randomization

```python
def _load_scene(self, options):
    loader = self.scene.create_urdf_loader()
    
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

    articulation_builders = loader.parse(str(urdf_path))["articulation_builders"]
    builder = articulation_builders[0]
    builder.build(name="my_articulation")
```

### Articulation Limitations

For the PhysX simulation backend, any single articulation can have a maximum of 64 links. More complex articulated objects will need to be simplified by merging links together. Most of the time this is readily possible by inspecting the URDF and fusing together links held together by fixed joints. The less fixed joints and links there are, the better the simulation will run in terms of accuracy and speed.

## Using the MJCF Loader

If your actor/articulation is defined with a MJCF file, you can use a MJCF loader to load that articulation and make modifications as needed. It works the exact same as the [URDF loader](./loading_objects.md#using-the-urdf-loader). Note that not all properties in MJCF/MuJoCo are supported in SAPIEN/ManiSkill at this moment, so you should always verify your articulation/actors are loaded correctly from the MJCF.

```python
def _load_scene(self, options):
    loader = self.scene.create_mjcf_loader()
    builders = loader.parse(str(mjcf_path))
    articulation_builders = builders["articulation_builders"]
    actor_builders = builders["actor_builders"]
```

## Querying Meshes and Bounds

The returned `Actor` and `Articulation` objects expose helper methods for fetching collision meshes as `trimesh.Trimesh` objects. `Articulation` also exposes visual-mesh helpers. These methods are useful for computing object heights, bounding boxes, visualization geometry, or placement offsets. Actor collision meshes can be queried immediately after the actor is built.

```python
def _load_scene(self, options):
    # Build or load self.object first.
    collision_mesh = self.object.get_first_collision_mesh(to_world_frame=False)
    if collision_mesh is not None:
        object_height = collision_mesh.bounding_box.extents[2]
```

Use `to_world_frame=False` when you want dimensions in the object's local frame, for example to compute how high to place an object above a table. Use `to_world_frame=True` when you need the mesh at its current world pose.

During GPU simulation, all articulation collision and visual mesh queries require GPU simulation to be initialized, including queries with `to_world_frame=False`. Do not query articulation meshes in `_load_scene`; query them after initialization instead.

## Reconfiguring and Optimization

In general loading is always quite slow, especially on the GPU so by default, ManiSkill reconfigures just once. Any call to `env.reset()` will not trigger a reconfiguration unless you call `env.reset(seed=seed, options=dict(reconfigure=True))` (seed is not needed but recommended if you are reconfiguring for reproducibility). 

However, during CPU simulation with just a single environment (or GPU simulation with very few environments) the loaded object geometries never get to change as reconfiguration doesn't happen more than once. This behavior can be changed by setting the `reconfiguration_freq` value of your task. 

The recommended way to do this is as follows (taken from the PickSingleYCB task):

```python
class PickSingleYCBEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
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

A `reconfiguration_freq` value of 1 means during every reset we reconfigure. A `reconfiguration_freq` of `k` means every `k` resets we reconfigure. A `reconfiguration_freq` of 0 (the default) means we never reconfigure again.

In general one use case of setting a positive `reconfiguration_freq` value is for when you want to simulate a task in parallel where each parallel environment is working with a different object/articulation and there are way more object variants than number of parallel environments. For machine learning / RL workflows, setting `reconfiguration_freq` to e.g. 10 ensures every 10 resets the objects being simulated on are randomized which can diversify the data collected for online training while keeping simulation fast by reconfiguring infrequently.
