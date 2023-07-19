import gymnasium as gym
import pytest

from mani_skill2.envs.sapien_env import BaseEnv
import numpy as np
ENV_IDS = [
    "LiftCube-v0",
    "PickCube-v0",
    "StackCube-v0",
    "PickSingleYCB-v0",
    "PickSingleEGAD-v0",
    "PickClutterYCB-v0",
    "AssemblingKits-v0",
    "PegInsertionSide-v0",
    "PlugCharger-v0",
    "PandaAvoidObstacles-v0",
    "TurnFaucet-v0",
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    "PushChair-v1",
    "MoveBucket-v1",
]
OBS_MODES = [
    "state_dict",
    "state",
    "rgbd",
    "pointcloud",
    "rgbd_robot_seg",
    "pointcloud_robot_seg",
]

@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("obs_mode", OBS_MODES)
def test_envs(env_id, obs_mode):
    env = gym.make(env_id, obs_mode=obs_mode)
    env.reset()
    action_space = env.action_space
    for _ in range(5):
        env.step(action_space.sample())
    env.close()
    del env

def test_env_seeded_reset():
    env = gym.make(ENV_IDS[0])
    obs, _ = env.reset(seed=2000)
    for _ in range(5):
        env.step(env.action_space.sample())
    new_obs, _ = env.reset(seed=2000)
    np.testing.assert_allclose(obs, new_obs, atol=1e-4)
    
    env.reset()
    new_obs, _ = env.reset(seed=2000)
    np.testing.assert_allclose(obs, new_obs, atol=1e-4)
    env.close()
    del env