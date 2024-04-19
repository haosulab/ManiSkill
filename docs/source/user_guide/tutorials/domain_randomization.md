# Domain Randomization

One of the benefits of simulation is the ability to chop and change a number of aspects that would otherwise be expensive or time-consuming to do in the real world. This documents demonstrates a number of simple tools for randomization from texture randomization to camera pose randomizations.

## Camera Randomization

For cameras, which are created by adding `CameraConfig` objects to your task's `_default_sensor_configs` property, you can randomize the pose and fov across all parallel sub-scenes. This can be done either during reconfiguration or episode initialization.

### During Reconfiguration

Simply providing batched data to the CameraConfigs of your sensors as done below (pose, fov, near, and far are supported) will randomize the camera configuration across parallel scenes. The example below does it for camera poses by modifying the `_default_sensor_configs` property of the task class

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

It will generate the following result (for 16 parallel environments) on e.g. the PickCube-v1 task:

:::{figure} images/camera_domain_randomization.png
:::


Note that this method of randomization only randomizes during task reconfiguration, not during each episode reset (which calls `_initialize_episode`). In GPU simulation with enough parallel environments it shouldn't matter too much if you never reconfigure again, but if you wish you can set a `reconfiguration_freq` value documented [here](./custom_tasks.md#reconfiguring-and-optimization)

### During Episode Initialization / Resets

Cameras when created cannot have their configurations modified after reconfiguration. Thus it is not possible to randomize the camera's fov, near, and far configurations outside of reconfiguration. You can however still randomize the camera pose during resets via mounted cameras (albeit this is a little slower than doing just during reconfiguration). To get started, first in your `_load_scene` function you have to create an actor to represent the camera (which does not need any visual or collision shapes):

```python
def _load_scene(self, options: dict):
    # ... your loading code
    self.cam_mount = self._scene.create_actor_builder().build_kinematic("camera_mount")
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

Once the camera config is properly defined, we can move mount however we wish in the world and the camera will follow. For camera pose randomization you can copy the code written earlier to randomize camera pose directly into `_initialize_episode`

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