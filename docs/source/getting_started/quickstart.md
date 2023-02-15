# Quickstart

## Gym Interface

Here is a basic example of how to make a ManiSKill2 environment following the interface of [OpenAI Gym](https://github.com/openai/gym) and run a random policy.

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

The common observation modes are `state`, `rgbd`, `pointcloud`. We also support `state_dict` (states organized as a hierachical dictionary) and `image` (raw visual observations without postprocessing). Please refer to [Observation](../concepts/observation.md) for more details.

We support a wide range of controllers. Different controllers can have different effects on your algorithms. Thus, it is recommended to understand the action space you are going to use. Please refer to [Controllers](../concepts/controllers.md) for more details.

Some environments require **downloading assets**. You can download all the assets by `python -m mani_skill2.utils.download_asset all` or download task-specific assets by `python -m mani_skill2.utils.download_asset ${ENV_ID}`. Please refer to [Environments](../concepts/environments.md) for all supported environments, and which environments require downloading assets.

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

To enable an interactive viewer supported by SAPIEN, you can add `--enable-sapien-viewer`. The interactive SAPIEN viewer is more powerful for debugging (e.g., checking collision shapes, getting currrent poses). There will be two windows: an OpenCV window and a SAPIEN (GL) window. Pressing `0` on the focused window can switch the control to the other one.

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

## Tutorials

