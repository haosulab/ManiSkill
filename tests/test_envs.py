import gym
import pytest

from mani_skill2.envs.sapien_env import BaseEnv

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


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_envs(env_id):
    OBS_MODES = ["state_dict", "state", "rgbd", "pointcloud"]
    for obs_mode in OBS_MODES:
        env: BaseEnv = gym.make(env_id, obs_mode=obs_mode)
        env.reset()
        action_space = env.action_space
        for _ in range(5):
            env.step(action_space.sample())
        env.close()
        del env
