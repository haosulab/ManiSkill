# Quickstart

## Gym Interface

Here is a basic example of how to make a ManiSkill task following the interface of [Gymnasium](https://gymnasium.farama.org/) and run a random policy.

```python
import gymnasium as gym
import mani_skill2.envs

env = gym.make(
    "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    obs_mode="state", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    render_mode="human"
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0) # reset with a seed for determinism
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()  # a display is required to render
env.close()
```

You can also run the same code from the command line to demo random actions

```bash
python -m mani_skill2.examples.demo_random_action -e PickCube-v1 # run headless
python -m mani_skill2.examples.demo_random_action -e PickCube-v1 --render-mode="human" # run with A GUI
```

```{figure} images/demo_random_action_gui.png
---
alt: SAPIEN GUI showing the PickCube task
---
```

Each ManiSkill task supports different **observation modes** and **control modes**, which determine its **observation space** and **action space**. They can be specified by `gym.make(env_id, obs_mode=..., control_mode=...)`.

The common observation modes are `state`, `rgbd`, `pointcloud`. We also support `state_dict` (states organized as a hierarchical dictionary) and `image` (raw visual observations without postprocessing). Please refer to [Observation](../concepts/observation.md) for more details.

We support a wide range of controllers. Different controllers can have different effects on your algorithms. Thus, it is recommended to understand the action space you are going to use. Please refer to [Controllers](../concepts/controllers.md) for more details.

Some tasks require **downloading assets** that are not stored in the python package itself. You can download task-specific assets by `python -m mani_skill2.utils.download_asset ${ENV_ID}`. The assets will be downloaded to `~/maniskill/data` by default, but you can also use the environment variable `MS_ASSET_DIR` to change this destination. Please refer to [Tasks](../concepts/tasks.md) for all tasks built in out of the box, and which tasks require downloading assets.

We also have demos for simulations of scenes like ReplicaCAD, which can be run by doing

```bash
python -m mani_skill2.utils.download_asset "ReplicaCAD"
python -m mani_skill2.examples.demo_random_action.py -e "ReplicaCAD_SceneManipulation-v1" --render-mode="rgb_array" --record-dir="videos" # run headless and save video
python -m mani_skill2.examples.demo_random_action.py -e "ReplicaCAD_SceneManipulation-v1" --render-mode="human" # run with GUI (recommended!)
```



<video preload="auto" controls="True" width="100%">
<source src="/_static/videos/fetch_random_action_replica_cad_rt.mp4" type="video/mp4">
</video>

For more details on rendering see TODO (stao). For a compilation of demos you can run without having to write any extra code check out the [demos page](../demos/index)

## GPU Parallelized/Vectorized Tasks

ManiSkill is powered by SAPIEN which supports GPU parallelized physics simulation and GPU parallelized rendering. This enables achieving 200,000+ state-based simulation FPS and 10,000+ FPS with rendering on a single 4090 GPU. For full benchmarking results see [this page](../additional_resources/performance_benchmarking)

In order to run massively parallelized tasks on a GPU, it is as simple as adding the `num_envs` argument to `gym.make` as so

```python
import gymnasium as gym
import mani_skill2.envs

env = gym.make("PickCube-v1", num_envs=1024)
print(env.observation_space) # will now have shape (1024, ...)
print(env.action_space) # will now have shape (1024, ...)
```

To benchmark the parallelized simulation, you can run 

```bash
python -m mani_skill2.examples.benchmarking.gpu_sim --num-envs=1024
```

To try out the parallelized rendering, you can run

```bash
# rendering RGB + Depth data from all cameras
python -m mani_skill2.examples.benchmarking.gpu_sim --num-envs=64 --obs-mode="rgbd"
# directly save 64 videos of the visual observations put into one video
python -m mani_skill2.examples.benchmarking.gpu_sim --num-envs=64 --save-video
```