# GPU Simulation on SAPIEN Notes

some language:
sub-scene: refers to each individually created scene that is part of the larger scene (the actual physx scene). The sub-scenes
are now managed by a new ManiskillScene class, which exposes some useful API for building tasks under the assumption that all the tasks share the same actors/data shapes. Masking+Different objects per scene is TBD


body_data -> Refers to data for anything created with actor builders
- Must allocate for each actor created in each sub scene (sapien does not assume the same actor exists in all sub scenes)
- is a 16-dim flat vector. :7 is Pose (quaternion followed by position), :6 is velocities (linear followed by angular)

can also retrieve raw body data, which is in the physx format. which means :4 is quaternion, 4:7 is position, 8:11 is linear velocity, 

```
print("quat", data_buffer_raw[:2, :4], pos", data_buffer_raw[:2, 4:7], "vel", data_buffer[:2, 8:11])
```

articulation_*_buffer -> Any articulation related data, this includes buffers for
- Robot configuration (q) values, controlling position (qpos), velocity (qvel), and force (qf)
- Robot root pose
- ...

Now static objects cannot change pose once built in a scene. In CPU simulation this was not enforced. in GPU simulation this now is. 

For objects that should be kinematic but can change pose (majority of our code as we often set initial poses after building) use build_kinematic


gpu_init is nearly equivalent to taking a single step in the simulation. It is unclear what it actually does and it might have some bugs. Upon creating a cube in two scenes, one cube is in the wrong pose...

## new gpu api

ok so gpu_pose_index is for the large buffer, gpu_index is for the sliced one. `physx.PhysxRigidDynamicComponent` and `physx.PhysxArticulationLinkComponent` have this property. Note static objects do not have this, user should maintain that data somewhere.

To update gpu data, you edit the buffers in place. This also directly updates the renderer so if u run update render and try to take an image you see the updates live. The simulation is not affected until an apply call is made.


## developing with GPU SAPIEN to support GPU and CPU work
We automatically wrap many of the common classes in SAPIEN/exposed physx API by either creating a new class that inherits the original class or making a new class that doesn't inherit but adds all the same functionality. The wrapper classes purpose is to allow managing multiple of the same represented object in SAPIEN across multiple sub-scenes. 

The wrapped Articulation class is a wrapper over the physx.PhysxArticulation class, and manages the same articulation across multiple sub-scenes. 

Some classes cannot use inheritance because the original class is part of the C++ code or some other tricky reasons (e.g. the wrapper class can't call the original class' init function )


In general, if one can inherit the original class, do it. If not, re-add all the functions. This way we have type hints as well. The alternative involves overriding `__setattr__` and `__getattr__` and is very messy + no type hints sometimes.

## GPU ManiSkill Notes

all _load functions are "single-thread" mode for the most part if you use the ManiSkillScene API (to create Articulation and Actor builders). You build once just like before, and we parallel create and manage the GPU tensors. These are made single-thread so that
- same code as before works. You build for one scene only if use the maniskill APIs (customizer per scene by using sapien apis)
- easy management of GPU tensors by keeping all on one object instead of doing a for loop and storing stuff yourself...

all _initialize functions however are NOT "single-thread".


Parallel IK?

cannot set collision groups in gpu sim after links are built...


## Legged Task Simulation

OmniIsaacGymEnvs:

They have anymal.py which tasks the anymal robot to move in a given direction and anymal_terrain.py is the same but on random complicated terrain

Some differences of note:
- anymal.py uses a joint delta position controller
- anymal_terrain.py sets torque efforts via
```
torques = torch.clip(
    self.Kp * (self.action_scale * self.actions + self.default_dof_pos - self.dof_pos)
    - self.Kd * self.dof_vel,
    -80.0,
    80.0,
)
```

They interestingly add a self.default_dof_pos - self.dof_pos which naturally pushes the robot to a resting position