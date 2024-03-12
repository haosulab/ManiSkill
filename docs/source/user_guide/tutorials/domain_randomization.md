# Domain Randomization

One of the benefits of simulation is the ability to chop and change a number of aspects that would otherwise be expensive or time-consuming to do in the real world. This documents demonstrates a number of simple tools for randomization from texture randomization to camera pose randomizations.

## Camera Randomization

For cameras, which are created by adding `CameraConfig` objects to your task's `_sensor_configs` property, you can randomize the pose and fov across all parallel sub-scenes. Simply provide batched data to the CameraConfig as done below for the poses.

```python
from mani_skill.envs.utils import randomization
from mani_skill.utils import sapien_utils
@property
def _sensor_configs(self):
    pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
    pose = Pose.create(pose)
    pose = pose * Pose.create_from_pq(
        p=torch.rand((self.num_envs, 3)) * 0.05 - 0.025,
        q=randomization.random_quaternions(
            n=self.num_envs, device=self.device, bounds=(-np.pi / 24, np.pi / 24)
        ),
    )
    return [CameraConfig("base_camera", pose, width=128, height=128, fov=np.pi / 2, near=0.01, far=100)]
```

It will generate the following result (for 16 parallel environments) on e.g. the PickCube-v1 task:

:::{figure} images/camera_domain_randomization.png
:::
