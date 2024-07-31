import gymnasium as gym
import pytest
import sapien
import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.structs.pose import Pose


def test_link_pose():
    env = PickCubeEnv()
    env.agent.robot.pose = sapien.Pose(p=[0.2, 0.3, 0.5])
    assert torch.isclose(env.agent.robot.pose.p[0], torch.tensor([0.2, 0.3, 0.5])).all()
    env.agent.robot.pose = torch.tensor([0.4, 0.5, 0.6, 1, 0, 0, 0])
    assert torch.isclose(env.agent.robot.pose.p[0], torch.tensor([0.4, 0.5, 0.6])).all()
    assert torch.isclose(env.agent.robot.pose.q[0], torch.tensor([1.0, 0, 0, 0])).all()
    env.agent.robot.pose = Pose.create(torch.tensor([0.2, 0.5, 0.6, 1, 0, 0, 0]))
    assert torch.isclose(env.agent.robot.pose.p[0], torch.tensor([0.2, 0.5, 0.6])).all()
    assert torch.isclose(env.agent.robot.pose.q[0], torch.tensor([1.0, 0, 0, 0])).all()


@pytest.mark.gpu_sim
def test_link_pose_gpu():
    env = PickCubeEnv(num_envs=4)
    with torch.device(env.device):
        env.agent.robot.pose = sapien.Pose(p=[0.2, 0.3, 0.5])
        assert torch.isclose(
            env.agent.robot.pose.p[0], torch.tensor([0.2, 0.3, 0.5])
        ).all()
        env.agent.robot.pose = torch.tensor([0.4, 0.5, 0.6, 1, 0, 0, 0])
        assert torch.isclose(
            env.agent.robot.pose.p[0], torch.tensor([0.4, 0.5, 0.6])
        ).all()
        assert torch.isclose(
            env.agent.robot.pose.q[0], torch.tensor([1.0, 0, 0, 0])
        ).all()
        env.agent.robot.pose = Pose.create(torch.tensor([0.2, 0.5, 0.6, 1.0, 0, 0, 0]))
        for i in range(4):
            assert torch.isclose(
                env.agent.robot.pose.p[i], torch.tensor([0.2, 0.5, 0.6])
            ).all()
            assert torch.isclose(
                env.agent.robot.pose.q[i], torch.tensor([1.0, 0, 0, 0])
            ).all()
