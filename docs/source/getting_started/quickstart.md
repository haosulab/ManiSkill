# Quickstart

## Gym Interface

Here is a basic example of how to make a ManiSkill2 environment following the interface of [OpenAI Gym](https://github.com/openai/gym) and run a random policy.

```python
import gym
import mani_skill2.envs  # import to register all environments in gym

env = gym.make("PickCube-v0", obs_mode="rgbd", control_mode="pd_ee_delta_pose")
print("Observation space", env.observation_space)
print("Action space", env.action_space)

env.seed(0)  # specify a seed for randomness
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()  # a display is required to render
env.close()
```

Each ManiSkill2 environment supports different **observation modes** and **control modes**, which determine its **observation space** and **action space**. They can be specified by `gym.make(env_id, obs_mode=..., control_mode=...)`.

The common observation modes are `state`, `rgbd`, `pointcloud`. We also support `state_dict` (states organized as a hierarchical dictionary) and `image` (raw visual observations without postprocessing). Please refer to [Observation](../concepts/observation.md) for more details.

We support a wide range of controllers. Different controllers can have different effects on your algorithms. Thus, it is recommended to understand the action space you are going to use. Please refer to [Controllers](../concepts/controllers.md) for more details.

Some environments require **downloading assets**. You can download all the assets by `python -m mani_skill2.utils.download_asset all` or download task-specific assets by `python -m mani_skill2.utils.download_asset ${ENV_ID}`. The assets will be downloaded to `./data/` by default, and you can also use the environment variable `MS2_ASSET_DIR` to specify this destination. Please refer to [Environments](../concepts/environments.md) for all supported environments, and which environments require downloading assets.

## Interactive Play

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

## Vectorized Environment

We provide an implementation of vectorized environments (for rigid-body environments) optimized for visual observations, powered by the SAPIEN RPC-based render server-client system. Importantly, our vectorized environment parallelizes rendering (visual observations) and computing non-visual observations as well as rewards, in a communication-efficient way (visual observation tensors are kept on the GPU to avoid unnecessary data transfer).

It is easy to create a vectorized environment:

```python
from mani_skill2.vector import VecEnv, make
env: VecEnv = make("PickCube-v0", num_envs=4)
```

Please see `mani_skill2/examples/demo_vec_env.py` for a complete example:

```bash
python -m mani_skill2.examples.demo_vec_env -e PickCube-v0 -n 4
```

We provide examples to use our `VecEnv` with [Stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/). Please refer to our [notebook](https://github.com/haosulab/ManiSkill2/blob/main/examples/tutorials/2_reinforcement_learning.ipynb) or [example scripts](https://github.com/haosulab/ManiSkill2/tree/main/examples/tutorials/reinforcement-learning).

---

**Implementation details**: The vectorized environment is optimized for visual observations. In short, the vectorized environment creates multiple python processes (workers) to run the physical simulation for each environment. For each timestep, each worker will compute non-visual observations and rewards in parallel with rendering visual observations. Specifically, the worker (client) sends information needed for rendering to the main process (server), and the actual work of rendering is done by the server. Thus, non-visual and visual observations are obtained in parallel, and the amount of information to communicate between processes is minimized.

:::{note}
- The vectorized environment only supports observation modes including visual observations (`rgbd`, `pointcloud`, `image`). If only state observations are needed, most RL libraries (like Stable-baselines3) provide their implementations of multi-process vectorized environments.
- The visual observations (rendered from cameras) are `torch.Tensor` while non-visual observations are `numpy.ndarray`. It is critical to keep tensors on the GPU for overall efficiency.
- `env.render()` is not supported in the vectorized environment. We suggest that you only use our implementation of vectorized environments for training.
:::

## Tutorials

We provide hands-on tutorials about ManiSkill2. All the tutorials can be found [here](https://github.com/haosulab/ManiSkill2/blob/main/examples/tutorials).

- Getting Started: [Jupyter Notebook](https://github.com/haosulab/ManiSkill2/blob/main/examples/tutorials/1_quickstart.ipynb), [Colab](https://colab.research.google.com/github/haosulab/ManiSkill2/blob/main/examples/tutorials/1_quickstart.ipynb)
- Reinforcement Learning: [Jupyter Notebook](https://github.com/haosulab/ManiSkill2/blob/main/examples/tutorials/2_reinforcement_learning.ipynb), [Colab](https://colab.research.google.com/github/haosulab/ManiSkill2/blob/main/examples/tutorials/2_reinforcement_learning.ipynb)
- Imitation Learning: [Jupyter Notebook](https://github.com/haosulab/ManiSkill2/blob/main/examples/tutorials/3_imitation_learning.ipynb), [Colab](https://colab.research.google.com/github/haosulab/ManiSkill2/blob/main/examples/tutorials/3_imitation_learning.ipynb)
- Environment Customization: [Jupyter Notebook](https://github.com/haosulab/ManiSkill2/blob/main/examples/tutorials/customize_environments.ipynb), [Colab](https://colab.research.google.com/github/haosulab/ManiSkill2/blob/main/examples/tutorials/customize_environments.ipynb)
- Advanced Rendering (ray tracing, stereo depth sensor): [Jupyter Notebook](https://github.com/haosulab/ManiSkill2/blob/main/examples/tutorials/advanced_rendering.ipynb)

ManiSkill2 is based on SAPIEN. SAPIEN tutorials are [here](https://sapien.ucsd.edu/docs/latest/).
