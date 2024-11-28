# Sensors / Cameras

This page documents how to use / customize sensors and cameras in ManiSkill in depth at runtime and in task/environment definitions. In ManiSkill, sensors are "devices" that can capture some modality of data. At the moment there is only the Camera sensor type.

## Cameras

Cameras in ManiSkill can capture a ton of different modalities/textures of data. By default ManiSkill limits those to just `rgb`, `depth`, `position` (which is used to derive depth), and `segmentation`. Internally ManiSkill uses [SAPIEN](https://sapien.ucsd.edu/) which has a highly optimized rendering system that leverages shaders to render different modalities of data. The full set of configurations for cameras can be found in {py:class}`mani_skill.sensors.camera.CameraConfig`.

Each shader has a preset configuration that generates image-like data, often in a somewhat difficult to use format due to heavy optimization. ManiSkill uses a shader configuration system in python that parses these different shaders into more user friendly texture formats like `rgb` and `depth`. See the [next section on the shaders](#shaders-and-textures) for more details about what textures are available to generate for which shader and their default shape/types.


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


### Shaders and Textures

The following shaders are available in ManiSkill:

| Shader Name | Available Textures                                 | Description                                                                                                                                     |
| ----------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| minimal     | rgb, depth, position, segmentation                 | The fastest shader with minimal GPU memory usage. Note that the background will always be black (normally it is the color of the ambient light) |
| default     | rgb, depth, position, segmentation, normal, albedo | A balance between speed and texture availability                                                                                                |
| rt          | rgb, depth, position, segmentation, normal, albedo | A shader optimized for photo-realistic rendering via ray-tracing                                                                                |
| rt-med      | rgb, depth, position, segmentation, normal, albedo | Same as rt but runs faster with slightly lower quality                                                                                          |
| rt-fast     | rgb, depth, position, segmentation, normal, albedo | Same as rt-med but runs faster with slightly lower quality                                                                                      |



The following textures are available in ManiSkill. Note all data is not scaled/normalized unless specified otherwise.

| Texture | Shape | dtype | Description |
|---------|-------|-------|-------------|
| rgb | [H, W, 3] | torch.uint8 | Red, Green, Blue colors of the image. Range of 0-255 |
| depth | [H, W, 1] | torch.int16 | Depth in millimeters |
| position | [H, W, 4] | torch.int16 | x, y, z in millimeters and 4th channel is same as segmentation below |
| segmentation | [H, W, 1] | torch.int16 | Segmentation mask with unique integer IDs for each object |
| normal | [H, W, 3] | torch.float32 | x, y, z components of the normal vector |
| albedo | [H, W, 3] | torch.uint8 | Red, Green, Blue colors of the albedo. Range of 0-255 |

