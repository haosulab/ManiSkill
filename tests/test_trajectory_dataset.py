import json

import h5py
import numpy as np
import pytest

from mani_skill.trajectory.dataset import ManiSkillTrajectoryDataset


def create_trajectory_dataset_file(path):
    """Create a small synthetic trajectory .h5/.json pair where success/fail are
    independent from terminated/truncated, like a continuing rollout that
    succeeds before reaching the time limit."""
    with h5py.File(path, "w") as f:
        traj = f.create_group("traj_0")
        traj["obs"] = np.zeros((4, 2), dtype=np.float32)
        traj["actions"] = np.zeros((3, 1), dtype=np.float32)
        traj["terminated"] = np.array([False, False, False])
        traj["truncated"] = np.array([False, False, True])
        traj["success"] = np.array([False, True, True])
        traj["fail"] = np.array([False, False, False])
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "env_info": {"env_id": "PickCube-v1", "env_kwargs": {}},
                "episodes": [{"episode_id": 0, "success": True}],
            }
        )
    )


EXPECTED_SUCCESS = [False, True, True]
EXPECTED_FAIL = [False, False, False]
EXPECTED_TERMINATED = [False, False, False]
EXPECTED_TRUNCATED = [False, False, True]


@pytest.mark.parametrize("device", [None, "cpu"])
def test_trajectory_dataset_preserves_success_fail_labels(tmp_path, device):
    """https://github.com/haosulab/ManiSkill/issues/1499

    The loader must preserve the recorded success/fail arrays regardless of
    whether tensor storage is requested via ``device``.
    """
    h5_path = tmp_path / "trajectory.h5"
    create_trajectory_dataset_file(h5_path)

    dataset = ManiSkillTrajectoryDataset(str(h5_path), device=device)
    try:
        assert dataset.success.tolist() == EXPECTED_SUCCESS
        assert dataset.fail.tolist() == EXPECTED_FAIL
        assert dataset.terminated.tolist() == EXPECTED_TERMINATED
        assert dataset.truncated.tolist() == EXPECTED_TRUNCATED
    finally:
        dataset.data.close()
