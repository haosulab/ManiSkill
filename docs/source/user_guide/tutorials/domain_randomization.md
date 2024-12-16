# Domain Randomization

One of the benefits of simulation is the ability to chop and change a number of aspects that would otherwise be expensive or time-consuming to do in the real world. This documents demonstrates a number of simple tools for randomization. In the beta release we currently only have a tutorial on how to do camera randomization.

## Camera Randomization

For cameras, which are created by adding `CameraConfig` objects to your task's `_default_sensor_configs` property, you can randomize the pose and fov across all parallel sub-scenes. This can be done either during reconfiguration or episode initialization.

### During Reconfiguration

Simply providing batched data to the CameraConfigs of your sensors as done below (pose, intrinsic, fov, near, and far are supported) will randomize the camera configuration across parallel scenes. The example below does it for camera poses by modifying the `_default_sensor_configs` property of the task class

```python
import torch
from mani_skill.envs.utils import randomization
from mani_skill.utils import sapien_utils

class MyCustomTask(BaseEnv):
    # ...
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        pose = Pose.create(pose)
        pose = pose * Pose.create_from_pq(
            p=torch.rand((self.num_envs, 3)) * 0.05 - 0.025,
            q=randomization.random_quaternions(
                n=self.num_envs, device=self.device, bounds=(-np.pi / 24, np.pi / 24)
            ),
        )
        return [CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2, near=0.01, far=100)]
```

To verify it works you can run the test script here

:::{dropdown} Test Script
```python
from mani_skill.utils.wrappers import RecordEpisode
from your_env_code import MyCustomTask
env = MyCustomTask(num_envs=16, render_mode="sensors")
env = RecordEpisode(env, "./videos", save_trajectory=False)
env.reset(seed=0)
for _ in range(10):
    env.step(env.action_space.sample())
env.close()
```
:::

It will generate the following result (for 16 parallel environments) on e.g. the PickCube-v1 task:

:::{figure} images/camera_domain_randomization.png
:::


Note that this method of randomization only randomizes during task reconfiguration, not during each episode reset (which calls `_initialize_episode`). In GPU simulation with enough parallel environments it shouldn't matter too much if you never reconfigure again, but if you wish you can set a `reconfiguration_freq` value documented [here](./custom_tasks/loading_objects.md#reconfiguring-and-optimization)

### During Episode Initialization / Resets

Cameras when created cannot have their configurations modified after reconfiguration. Thus it is not possible to randomize the camera's fov, near, and far configurations outside of reconfiguration. You can however still randomize the camera pose during resets via mounted cameras (albeit this is a little slower than doing just during reconfiguration). To get started, first in your `_load_scene` function you have to create an actor to represent the camera (which does not need any visual or collision shapes):

```python
def _load_scene(self, options: dict):
    # ... your loading code
    self.cam_mount = self.scene.create_actor_builder().build_kinematic("camera_mount")
```

Now we can mount our camera by modifying the camera config to specify where to mount it via the `mount` argument. Note that the pose argument of the camera config is now the local pose, and we simply use `sapien.Pose()` which gives the identity pose. The world pose of mounted cameras are equal to the pose of mount multiplied by the local pose, meaning we can easily move this camera around during episode initialization instead of during reconfiguration.

```python
import sapien
from mani_skill.sensors.camera import CameraConfig
class MyCustomTask(BaseEnv):
    # ...
    @property
    def _default_sensor_configs(self):
        return [
            CameraConfig(
                "base_camera", pose=sapien.Pose(), width=128, height=128, 
                fov=np.pi / 2, near=0.01, far=100, 
                mount=self.cam_mount
            )
        ]
```

Once the camera config is properly defined, we can move the camera mount however we wish in the world and the camera will follow. For camera pose randomization you can copy the code written earlier to randomize camera pose directly into `_initialize_episode`

```python
import torch
from mani_skill.envs.utils import randomization
from mani_skill.utils import sapien_utils
class MyCustomTask(BaseEnv):
    # ...
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # ...
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        pose = Pose.create(pose)
        pose = pose * Pose.create_from_pq(
            p=torch.rand((self.num_envs, 3)) * 0.05 - 0.025,
            q=randomization.random_quaternions(
                n=self.num_envs, device=self.device, bounds=(-np.pi / 24, np.pi / 24)
            ),
        )
        self.cam_mount.set_pose(pose)
```

The result is the same as during reconfiguration, but instead every episode reset the camera poses all get randomized. You can also easily add moving cameras that follow the robot itself or move on their own in this manner.

## Actor/Link Physical and Visual Randomizations

By default ManiSkill provides some convenient APIs that accept batched inputs for randomizing some properties of {py:class}`mani_skill.utils.structs.Actor` or {py:class}`mani_skill.utils.structs.Link` objects. In general if you want to randomize properties more granularly or if there isn't a convenience API available, you can do so by modifying each actor/link's components individually. Note at this granular level all values are expected to be python primitives or numpy based, torch tensors are not used. Normally these randomizations should be done during the `_load_scene` function in custom tasks.


```python
import numpy as np
from sapien.physx import PhysxRigidBodyComponent
from sapien.render import RenderBodyComponent

def _load_scene(self, options: dict):
    # ... other code for loading objects
    actor: Actor | Link
    for obj in actor._objs:
        # modify some property of obj's components here, which is one of the actors/links 
        # managed in parallel by the `actor` object
        
        # modifying physical properties
        rigid_body_component: PhysxRigidBodyComponent = obj.find_component_by_type(PhysxRigidBodyComponent)
        rigid_body_component.mass = 1.0
        rigid_body_component.angular_damping = 20.0
        # modify per collision shape properties
        for shape in obj.collision_shapes:
            shape.physical_material.dynamic_friction = 0.2
            shape.physical_material.static_friction = 0.4
            shape.physical_material.restitution = 0.2
            shape.density = 1000
            shape.patch_radius = 0.1
            shape.min_patch_radius = 0.1
        
        # modifying visual properties. NOTE that you may need to build
        # each actor separately instead of all together in each parallel and then merge them to do the following
        render_body_component: RenderBodyComponent = obj.find_component_by_type(RenderBodyComponent)
        for render_shape in render_body_component.render_shapes:
            for part in render_shape.parts:
                # you can change color, use texture files etc.
                part.material.set_base_color(np.array([255, 255, 255, 255]) / 255)
                # note that textures must use the sapien.render.RenderTexture2D 
                # object which allows passing a texture image file path
                part.material.set_base_color_texture(None)
                part.material.set_normal_texture(None)
                part.material.set_emission_texture(None)
                part.material.set_transmission_texture(None)
                part.material.set_metallic_texture(None)
                part.material.set_roughness_texture(None)
```

Similarly joints can also be modified in the same manner by iterating over each the `._objs` list property of {py:class}`mani_skill.utils.structs.ArticulationJoint` objects.

Note that during GPU simulation most physical properties must be set in an environment during the `_load_scene` function which runs before the GPU simulation initialization. Once the GPU simulation is initialized, some properties are fixed and can only be changed again if the environment is reconfigured.

Also note that in GPU simulation, due to rendering optimizations changing the visual property of one actor/link may also change the properties of other actors/links that share the same render material object. To fix this you should build each actor separately instead of all together in each parallel environment using scene masks/idxs (see [Scene Masks](./custom_tasks/advanced.md#scene-masks)) with a different color / `sapien.render.RenderMaterial` object. After doing so then merge the parallel actors to manage them as one object. You can then use the code above for randomizing visual properties

Example of visual randomizations of object colors is shown below for the PushT task.

:::{figure} images/color_domain_randomizations.png
:::
## Agent/Robot and Controller Randomizations

Agents/Robots and Controllers are abstractions around articulated objects to make it easy to swap controllers and load robots into scenes. Under the hood they have the same components as any other articulation.

Controller randomizations revolve around joint properties like damping/stiffness and friction values. To change each joint in each parallel environment at a granular level you can do the following:

For example in a custom environment when you load the scene
```python
def _load_scene(self, options: dict):
    for joint in self.agent.robot.joints:
        for obj in joint._objs:
            obj.set_drive_properties(stiffness=1000, damping=100, force_limit=1000)
            obj.set_friction(friction=0.5)
```
The controller randomizations can also be done on the fly after the GPU simulation has initialized (e.g. during the `_initialize_episode` function). Some controllers may have specific functionalities that can be changed on the fly as well. You can access the currently used controller object of an environment via `env.agent.controller` and modify that.

Other randomizations of the agent/robot outside of controllers revolve around the robot links itself (e.g. gripper frictions) you can do the following

```python
def _load_scene(self, options: dict):
    # iterate over every link in the robot and each managed parallel link and modify the collision shape materials
    # accordingly. Some examples are shown below.
    for link in self.agent.robot.links:
        for obj in link._objs:
            for shape in obj.collision_shapes:
                shape.physical_material.dynamic_friction = 0.2
                shape.physical_material.static_friction = 0.4
                shape.physical_material.restitution = 0.2
                shape.density = 1000
                shape.patch_radius = 0.1
                shape.min_patch_radius = 0.1
                
```

Note you can do in place changes to physical material properties like above, we do not recommend creating a new PhysxMaterial object as there is a limit to the number of PhysxMaterials that can be created (64K).

## Lighting/Rendering Randomizations

You can also modify overall lighting properties of the scene by overriding the default lighting loaded in the task's `_load_lighting` function. By default ManiSkill adds ambient light, and two directional lights with shadows disabled. The code below shows an example randomizing just the ambient light applied to the PickCube task during GPU simulation and rendering (using the "default" shader pack).

```python
import numpy as np
def _load_lighting(self, options: dict)
    for scene in self.scene.sub_scenes:
        scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
        scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
        scene.add_directional_light([0, 0, -1], [1, 1, 1])
```

<div style="display: flex; justify-content: center;">
<video controls="True" width="100%">
<source src="../../_static/videos/dr_lighting.mp4" type="video/mp4">
</video>
</div>
<br/>

Note that cameras added to tasks by default use the ["minimal" shader pack](../concepts/sensors.md#shaders-and-textures) which has a pure black background. You may want to use a more advanced shader like the "default" shader which does render the background by adding `shader_pack="default"` to the camera config to more visibly see the randomization effects.

## Background / Segmentation based Randomizations


It is common to randomize the "background" or specific objects of a scene for domain randomization. There are two ways in which you can do this. One is via the task building APIs in ManiSkill, the other is do apply green-screening to outputted images by ManiSkill environments based on segmentation masks.

### Via Task Building APIs

Without green-screening you can simply modify object textures directly in the `_load_scene` function of your custom task. See the section above on [Actor/Link Visual Randomizations](#actorlink-physical-and-visual-randomizations) for more details. You can change color, textures and more. This is a simple and flexible way to randomize object textures as you do not need to deal with finding segmentation masks.

### Green Screening

ManiSkill environments provide access to segmentation mask IDs via the `per_scene_id` of {py:class}`mani_skill.utils.structs.Actor` and {py:class}`mani_skill.utils.structs.Link` objects. Note that the ID 0 is reserved for the background. After collecting the IDs of all entities you wish to green-screen out with an image/video, you need to generate a segmentation mask in addition to the image data.

For sensor observations this can be done by ensuring you are using any observation mode that includes "segmentation" in the, which can be done by changing the `obs_mode` argument of `gym.make`. For example "rgb+segmentation" will output both RGB and segmentation images in environment observations returned by calls to `env.reset` and `env.step`. 

Example code segmenting out the background and overlaying a green screen image is shown below.

```python
import mani_skill.envs
import gymnasium as gym
import torch
import cv2
import matplotlib.pyplot as plt
env = gym.make("PickCube-v1", obs_mode="rgb+segmentation")
# get obs from reset
obs, _ = env.reset()
# get obs from step
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
your_image = cv2.resize(cv2.imread("path/to/your/image"), (128, 128))
green_screen_image = torch.from_numpy(your_image).to(device=env.device)
# only green-screen out the background and floor/ground in this case
seg_ids = torch.tensor([0], dtype=torch.int16, device=env.device)
seg_ids = torch.concatenate([seg_ids, env.unwrapped.scene.actors["ground"].per_scene_id])
for cam_name in obs["sensor_data"].keys():
    camera_data = obs["sensor_data"][cam_name]
    seg = camera_data["segmentation"]
    mask = torch.zeros_like(seg)
    mask[torch.isin(seg, seg_ids)] = 1
    camera_data["rgb"] = camera_data["rgb"] * (1 - mask) + green_screen_image * mask
    plt.imshow(camera_data["rgb"].cpu().numpy()[0])
    plt.show()
```
