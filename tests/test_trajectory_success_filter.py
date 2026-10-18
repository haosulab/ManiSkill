"""Check selected trajectory order and public observations across storage modes."""

import json

import h5py
import numpy as np
import pytest
import torch

from mani_skill.trajectory.dataset import ManiSkillTrajectoryDataset


@pytest.mark.parametrize(
    "device",
    [
        None,
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu_sim,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA storage requires a GPU"
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize("nested_obs", [False, True])
@pytest.mark.parametrize("success_only", [False, True])
@pytest.mark.parametrize("first_success", [False, True])
def test_first_retained_trajectory_and_appending(
    tmp_path, device, nested_obs, success_only, first_success
):
    path = tmp_path / "trajectory.h5"
    successes = [first_success, not first_success, True]
    observations = [np.arange(8).reshape(4, 2) + i * 100 for i in range(3)]
    markers = [np.arange(4, dtype=np.uint16).reshape(4, 1) + i * 10 for i in range(3)]
    with h5py.File(path, "w") as file:
        for i in range(3):
            trajectory = file.create_group(f"traj_{i}")
            if nested_obs:
                obs = trajectory.create_group("obs")
                obs.create_group("agent")["qpos"] = observations[i]
                obs.create_group("extra")["marker"] = markers[i]
            else:
                trajectory["obs"] = observations[i]
            trajectory["actions"] = np.full((3, 1), i)
            trajectory["terminated"] = [False, False, False]
            trajectory["truncated"] = [False, False, True]
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "env_info": {"env_id": "PickCube-v1", "env_kwargs": {}},
                "episodes": [
                    {"episode_id": i, "elapsed_steps": 3, "success": success}
                    for i, success in enumerate(successes)
                ],
            }
        )
    )
    selected = [i for i, success in enumerate(successes) if success or not success_only]
    expected_obs = np.concatenate([observations[i][:3] for i in selected])
    expected_actions = [[i] for i in selected for _ in range(3)]
    dataset = ManiSkillTrajectoryDataset(
        str(path), success_only=success_only, device=device
    )
    try:
        assert len(dataset) == len(expected_actions)
        rows = [dataset[i] for i in range(len(dataset))]
        if device is not None:
            assert all(row["action"].device.type == device for row in rows)
            leaves = [
                row["obs"]["agent"]["qpos"] if nested_obs else row["obs"]
                for row in rows
            ]
            assert all(leaf.device.type == device for leaf in leaves)
        assert [row["action"].tolist() for row in rows] == expected_actions
        if nested_obs:
            assert [
                row["obs"]["agent"]["qpos"].tolist() for row in rows
            ] == expected_obs.tolist()
            expected_markers = np.concatenate([markers[i][:3] for i in selected])
            assert [
                row["obs"]["extra"]["marker"].tolist() for row in rows
            ] == expected_markers.tolist()
            expected_dtype = np.dtype(np.int32) if device is None else torch.int32
            assert dataset.obs["extra"]["marker"].dtype == expected_dtype
        else:
            assert [row["obs"].tolist() for row in rows] == expected_obs.tolist()
        assert dataset.terminated.tolist() == [False] * len(expected_actions)
        assert dataset.truncated.tolist() == [False, False, True] * len(selected)
    finally:
        dataset.data.close()
