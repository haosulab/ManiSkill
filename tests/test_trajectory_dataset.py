"""Check that trajectory outcomes survive conversion to a requested device."""

import json

import h5py
import numpy as np
import pytest

from mani_skill.trajectory.dataset import ManiSkillTrajectoryDataset


@pytest.fixture
def outcome_trajectory(tmp_path):
    path = tmp_path / "trajectory.h5"
    outcomes = {
        "success": np.array([False, True, True]),
        "fail": np.array([False, False, False]),
        "terminated": np.array([False, False, False]),
        "truncated": np.array([False, False, True]),
    }
    with h5py.File(path, "w") as file:
        trajectory = file.create_group("traj_0")
        trajectory.create_dataset("obs", data=np.arange(8).reshape(4, 2))
        trajectory.create_dataset("actions", data=np.zeros((3, 1)))
        for name, values in outcomes.items():
            trajectory.create_dataset(name, data=values)
    metadata = {
        "env_info": {"env_id": "PickCube-v1", "env_kwargs": {}},
        "episodes": [{"episode_id": 0, "elapsed_steps": 3, "success": True}],
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return str(path), outcomes


@pytest.mark.parametrize("device", [None, "cpu"])
@pytest.mark.parametrize("outcome", ["success", "fail"])
def test_device_conversion_preserves_trajectory_outcomes(
    outcome_trajectory, device, outcome
):
    path, expected = outcome_trajectory
    dataset = ManiSkillTrajectoryDataset(path, device=device)
    try:
        actual = [dataset[index][outcome].item() for index in range(len(dataset))]
        assert actual == expected[outcome].tolist()
        assert dataset.terminated.tolist() == expected["terminated"].tolist()
        assert dataset.truncated.tolist() == expected["truncated"].tolist()
    finally:
        dataset.data.close()
