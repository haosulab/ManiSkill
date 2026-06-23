# Migration guide from ManiSkill 3 to ManiSkill 4

A guide for migrating code from ManiSkill 3 to ManiSkill 4, in addition to (strong) recommendations of things to be aware of when upgrading to ManiSkill 4, ranging from changes in default values like sim/control frequencies to TODO

## Things that you **must** change or be aware of

- There is no guarantee any randomness / RNG in maniskill 3 will return the same expected randomness in maniskill 4.

- State in maniskill 4 now includes joint targets and joint target velocities if available. Previously these were not included, which can sometimes lead to differences when setting state and taking `None` actions.

- Simulation and rendering backends are not limited to those provided by [Sapien](https://github.com/haosulab/sapien). Going forward, all backends are named with the following format: `<package_name:backend_name>`. So `physx_cuda` is now `sapien.physx_cuda` and the rendering backend `cuda` is now `sapien.cuda`. With the newton support since mujoco warp is used via newton, the mujoco warp backend is named `newton.mujoco_warp`. For a full list of possible backends and their details see TODO(stao). We will still provide backwards compatability however. Device IDs can still be specified as before e.g. `physx_cuda:0` becomes `sapien.physx_cuda:0` to run physx simulation on the `cuda:0` device.

- `mani_skill.sensors` module has been moved to `mani_skill.sim.sensors` instead. Moreover the StereoDepthSensor class is deprecated.

- There is no `px` field anymore in `self.scene`, a ManiSkillScene object. Calls to `self.scene.px.gpu_update_articulation_kinematics()` should change to `self.scene.physics_sim._gpu_update_articulation_kinematics()`

## Strong recommendations

- `SimConfig`, `DefaultMaterialsConfig` configs moved from `mani_skill/utils/structs/types.py` to `mani_skill/sim/base_sim.py`. They are now also frozen dataclasses meaning once created, you generally can't edit it (and shouldn't) unless you know what you are doing.

- The default control and simulation frequencies are now set to 60 and 120, matching a more typical ratio used in timekeeping (e.g. 60s per minute) and more typical values used in robotics for control frequencies. Previously they were 20 and 100, which is still a reasonable choice for many tasks when running simulation RL. Default maniskill tasks and baselines will update to be tuned for the 60 120 control/sim frequencies.

- Default tasks have a new environment ID version of `-v4` (to reflect maniskill v4). They are currently all `-v1`. I am not sure why we chose `-v1`.

- `Actor.px_body_type` is renamed to `Actor.body_type` since we support more than just physx.



## Full list of breaking changes including internal changes and deprecations not included in above


### Changes:

- To support multiple simulation backends beyond just SAPIEN, we now have a new general `BaseSim` class that defines all the standard functionalities of a simulator needed by ManiSkill, primarily physics (sub) stepping, rendering, and building objects/scenes with model builder type approaches as done by newton and SAPIEN. These simulator backend classes like `NewtonSim` and `SapienSim` are now located in `mani_skill/sim`. The hierarchcy of classes now from top (user interface) to bottom (sim/rendering code) is now mostly organized as `BaseEnv` > `ManiSkillScene` > `BaseSim`. While `BaseEnv` and `ManiSkillScene` both do not have simulator specific code anymore, we keep these two separate classes for organizational purposes. All robot learning, gymnasium-interface, and generic task building related code are in `BaseEnv`, with all code for managing different physics/rendering engines and exposing their data moved to `ManiSkillScene`.

- `BaseSim` class comes with a `BaseSimConfig` class which defines all the most typically important simulation configurations needed in ManiSkill that spread across backends. If a simulator backend doesn't support one of these configuration attributes (e.g. physics/simulation frequency), we probably will not support that simulator. If the attribute is not used (e.g. perhaps a GPU parallelized simulator has no concept of spacing if designed that way), that backend can simply choose to not parse that attribute. These configs are also more often than not frozen dataclasses now, making it harder to make the mistake of assuming some properties can be changed while a GPU sim is running.

### Deprecations

- `ManiSkillScene`, all sapien specific functions/properties are moved to `SapienSim`. Deprecate `timestep` property and setters/getters.
- `ActorBuilder` functions that add collisions no longer have a `patch_radius` or `min_patch_radius` option. These are physx specific and are only available if you specifically are using SAPIEN/physx backends. `is_trigger` is also no longer an argument, it was never used to begin with.
