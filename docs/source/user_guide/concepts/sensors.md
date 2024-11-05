# Sensors / Cameras

This page documents how to use / customize sensors and cameras in ManiSkill in depth at runtime and in task/environment definitions. In ManiSkill, sensors are "devices" that can capture some modality of data. At the moment there is only the Camera sensor type.

## Cameras

Cameras in ManiSkill can capture a ton of different modalities of data. By default ManiSkill limits those to just `rgb`, `depth`, `position` (which is used to derive depth), and `segmentation`. Internally ManiSkill uses [SAPIEN](https://sapien.ucsd.edu/) which has a highly optimized rendering system that leverages shaders to render different modalities of data. The full set of configurations can be found in {py:class}`mani_skill.sensors.camera.CameraConfig`.

Each shader has a preset configuration that generates textures containing data in a image format, often in a somewhat difficult to use format due to heavy optimization. ManiSkill uses a shader configuration system in python that parses these different shaders into more user friendly formats (namely the well known `rgb`, `depth`, `position`, and `segmentation` type data). This shader config system resides in this file on [Github](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/render/shaders.py) and defines a few friendly defaults for minimal/fast rendering and ray-tracing.


Every ManiSkill environment will have 3 categories of cameras (although some categories can be empty): sensors for observations for agents/policies, human_render_cameras for (high-quality) video capture for humans, and a single viewing camera which is used by the GUI application to render the environment.


At runtime when creating environments with `gym.make`, you can pass runtime overrides to any of these cameras as so. Below changes human render cameras to use the ray-tracing shader for photorealistic rendering, modifies sensor cameras to have width 320 and height 240, and changes the viewer camera to have a different field of view value.

```python
gym.make("PickCube-v1",
  sensor_configs=dict(width=320, height=240),
  human_render_camera_configs=dict(shader_pack="rt"),
  viewer_camera_configs=dict(fov=1),
)
```

These overrides will affect every camera in the environment in that group. So `sensor_configs=dict(width=320, height=240)` will change the width and height of every sensor camera in the environment, but will not affect the human render cameras or the viewer camera.

To override specific cameras, you can do it by camera name. For example, if you want to override the sensor camera with name `camera_0` to have a different width and height, you can do it as so:

```python
gym.make("PickCube-v1",
  sensor_configs=dict(camera_0=dict(width=320, height=240)),
)
```

Now all other sensor cameras will have the default width and height, and `camera_0` will have the specified width and height.

These specific customizations can be useful for those looking to customize how they render or generate policy observations to suit their needs. 