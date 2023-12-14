# Temporary Docs on environments with pre-built scenes

The code at the moment supports scenes created by [AI2-THOR](https://ai2thor.allenai.org/) stored in this Hugging Face Dataset: https://huggingface.co/datasets/hssd/ai2thor-hab/tree/main

Other scenes from other groups/projects are supportable provided you write an `SceneAdapter` class to load that type of configuration. See `mani_skill2/envs/scenes/adapters/hssd` for one such example.

To download ai2thor-hab scene dataset, in the root level run

```bash
# note that this is a very big dataset (~20GB) with 50,000+ files so it can take some time to download.
python mani_skill2/envs/scenes/adapters/hssd/download.py <your_hugging_face_access_token>
```

which downloads all the AI2-THOR scenes to `data/scene_datasets/ai2thor`

A dummy environment that uses just the ArchitecTHOR set of scenes (high quality human built scenes from AI2) can be created via standard gym as so

```python
import mani_skill2.envs
import gymnasium as gym
# PickObjectScene-v0 selects a scene randomly from the selected scene datasets and 
# instantiates a robot randomly and selects a random object for the robot to find and pick up.
# render_mode="human" opens up a viewer, convex_decomposition="none" makes scene loading fast (but not well simulated)
# set convex_decomposition="coacd" to use CoACD to get better collision meshes
import sapien.render
env = gym.make("PickObjectScene-v0", scene_datasets=["ArchitecTHOR"], render_mode="human", convex_decomposition="none", fixed_scene=True)

# optionally set these to make it more realistic
sapien.render.set_camera_shader_dir("rt")
sapien.render.set_viewer_shader_dir("rt")
sapien.render.set_ray_tracing_samples_per_pixel(4)
sapien.render.set_ray_tracing_path_depth(2)
sapien.render.set_ray_tracing_denoiser("optix")

env.reset(seed=2, options=dict(reconfigure=True))

while True:
    env.render()
```

Note that we set `fixed_scene=True` which is the default option. This means all calls to env.reset(seed=seed) or just env.reset() will always use the same scene. To change scene simply call `env.reset(seed=seed, options=dict(reconfigure=True))` which will always create the same scene depending on the seed here. When `fixed_scene=False` then every call to env.reset will create a new scene and the seed will dictate which scene is created.


## TODOs

- Find a better location to save metadata for scene datasets
- Pick reasonable initial states for robots in scenes that don't collide with the scene.
- More intuitive API for scene setting?
- Add in mobile robots