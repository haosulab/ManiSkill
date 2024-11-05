
import pytest
import torch
from mani_skill.envs.tasks.tabletop.stack_pyramid import StackPyramidEnv
from mani_skill.utils.structs.pose import Pose

import gymnasium as gym


import numpy as np

@pytest.fixture
def env():
    # return StackPyramidEnv()
    env = gym.make("StackPyramid-v1", obs_mode="state")
    return env

def test_env_initialization(env):
    assert env is not None

def test_env_reset(env):
    obs, info = env.reset()
    assert isinstance(obs, torch.Tensor)
    assert isinstance(info, dict)

def test_observation_space(env):
    _ = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

def test_action_space(env):
    action = env.action_space.sample()
    print(type(action))
    assert isinstance(action, np.ndarray)

def test_step(env):
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(type(obs), type(reward), type(terminated), type(truncated), type(info))
    assert isinstance(obs, torch.Tensor)
    assert isinstance(reward, torch.Tensor)
    assert isinstance(terminated, torch.Tensor)
    assert isinstance(truncated, torch.Tensor)
    assert isinstance(info, dict)

def test_success_condition(env):
    env.reset()
    # Manually set the positions of the cubes to a successful configuration
    env.cubeA.set_pose(Pose.create_from_pq(p=np.array([0.0, 0.0, 0.0])))
    env.cubeB.set_pose(Pose.create_from_pq(p=np.array([0.04, 0.0, 0.0])))
    env.cubeC.set_pose(Pose.create_from_pq(p=np.array([0.02, 0.0, 0.04])))
    env.cubeA.is_static = lambda lin_thresh, ang_thresh: True
    env.cubeB.is_static = lambda lin_thresh, ang_thresh: True
    env.cubeC.is_static = lambda lin_thresh, ang_thresh: True
    env.agent.is_grasping = lambda cube: False

    success_info = env.evaluate()
    # while True:
    #     env.render_human()

    print("Cube A Position:", env.cubeA.pose.p)
    print("Cube B Position:", env.cubeB.pose.p)
    print("Cube C Position:", env.cubeC.pose.p)
    print("Success Info:", success_info)


    assert success_info["success"], "Success condition failed."

