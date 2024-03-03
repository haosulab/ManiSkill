# Quickstart

## Gym Interface

Here is a basic example of how to make a ManiSkill2 environment following the interface of [Gymnasium](https://gymnasium.farama.org/) and run a random policy.

```python
import gymnasium as gym
import mani_skill2.envs

env = gym.make(
    "PickCube-v0", # there are more tasks e.g. "PushCube-v0", "PegInsertionSide-v0, ...
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

Each ManiSkill2 environment supports different **observation modes** and **control modes**, which determine its **observation space** and **action space**. They can be specified by `gym.make(env_id, obs_mode=..., control_mode=...)`.

The common observation modes are `state`, `rgbd`, `pointcloud`. We also support `state_dict` (states organized as a hierarchical dictionary) and `image` (raw visual observations without postprocessing). Please refer to [Observation](../concepts/observation.md) for more details.

We support a wide range of controllers. Different controllers can have different effects on your algorithms. Thus, it is recommended to understand the action space you are going to use. Please refer to [Controllers](../concepts/controllers.md) for more details.

Some environments require **downloading assets**. You can download all the assets by `python -m mani_skill2.utils.download_asset all` or download task-specific assets by `python -m mani_skill2.utils.download_asset ${ENV_ID}`. The assets will be downloaded to `./data/` by default, and you can also use the environment variable `MS2_ASSET_DIR` to specify this destination. Please refer to [Environments](../concepts/environments.md) for all supported environments, and which environments require downloading assets.

## Interactive Play

TODO (stao): Add demo of teleoperation from camera

We provide an example script to interactively play with our environments. A display is required.

```bash
# PickCube-v0 can be replaced with other environment id.
python -m mani_skill2.examples.demo_manual_control -e PickCube-v0
```

Keyboard controls:

- Press `i` (or `j`, `k`, `l`, `u`, `o`) to move the end-effector.
- Press any key between `1` to `6` to rotate the end-effector.
- Press `f` or `g` to open or close the gripper.
- Press `w` (or `a`, `s`, `d`) to translate the base if the robot is mobile. Press `q` or `e` to rotate the base. Press `z` or `x` to lift the torso.
- Press `esc` to close the viewer and exit the program.

To enable an interactive viewer supported by SAPIEN, you can add `--enable-sapien-viewer`. The interactive SAPIEN viewer is more powerful for debugging (e.g., checking collision shapes, getting current poses). There will be two windows: an OpenCV window and a SAPIEN (GL) window. Pressing `0` on the focused window can switch the control to the other one.

```{image} images/OpenCV-viewer.png
---
height: 256px
alt: OpenCV viewer
---
```

```{image} images/SAPIEN-viewer.png
---
height: 256px
alt: SAPIEN viewer
---
```

## GPU Parallelized/Vectorized Environments

ManiSkill is powered by SAPIEN which supports GPU parallelized physics simulation and GPU parallelized rendering. This enables achieving 200,000+ state-based simulation FPS and 10,000+ FPS with rendering on a single 4090 GPU. For full benchmarking results see [this page](../additional_resources/performance_benchmarking)

In order to run massively parallelized environments on a GPU, it is as simple as adding the `num_envs` argument to `gym.make` as so

```python
import gymnasium as gym
import mani_skill2.envs

env = gym.make("PickCube-v0", num_envs=1024)
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
python -m mani_skill2.examples.benchmarking.gpu_sim --num-envs=128 --obs-mode="rgbd"
# directly save 128 videos of the visual observations put into one video
python -m mani_skill2.examples.benchmarking.gpu_sim --num-envs=128 --save-video
```


<!-- 
We provide examples to use our `VecEnv` with [Stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/). Please refer to our [notebook](https://github.com/haosulab/ManiSkill2/blob/main/examples/tutorials/2_reinforcement_learning.ipynb) or [example scripts](https://github.com/haosulab/ManiSkill2/tree/main/examples/tutorials/reinforcement-learning). -->

<!-- ---

**Implementation details**: The vectorized environment is optimized for visual observations. In short, the vectorized environment creates multiple python processes (workers) to run the physical simulation for each environment. For each timestep, each worker will compute non-visual observations and rewards in parallel with rendering visual observations. Specifically, the worker (client) sends information needed for rendering to the main process (server), and the actual work of rendering is done by the server. Thus, non-visual and visual observations are obtained in parallel, and the amount of information to communicate between processes is minimized.

:::{note}
- The vectorized environment only supports observation modes including visual observations (`rgbd`, `pointcloud`, `image`). If only state observations are needed, most RL libraries (like Stable-baselines3) provide their implementations of multi-process vectorized environments.
- The visual observations (rendered from cameras) are `torch.Tensor` while non-visual observations are `numpy.ndarray`. It is critical to keep tensors on the GPU for overall efficiency.
- `env.render()` is not supported in the vectorized environment. We suggest that you only use our implementation of vectorized environments for training.
::: -->