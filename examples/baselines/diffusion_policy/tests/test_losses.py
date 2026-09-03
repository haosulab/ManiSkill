import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# The baseline is installed separately via its local setup.py.  Keep the
# regression tests runnable directly from a source checkout as well.
sys.path.insert(0, str(Path(__file__).parents[1]))

from diffusion_policy.losses import action_window_mask, masked_mse_loss


def test_action_window_mask_marks_only_executed_actions():
    # The window contains two pre-episode and two post-episode positions.
    mask = action_window_mask(start=-2, end=6, trajectory_length=4)
    assert mask.tolist() == [False, False, True, True, True, True, False, False]


def test_action_window_mask_covers_short_episode_without_empty_slice():
    # A short episode still yields a fixed-size mask with its three actions in
    # the correct trajectory-relative positions.
    mask = action_window_mask(start=-1, end=7, trajectory_length=3)
    assert mask.tolist() == [False, True, True, True, False, False, False, False]


def test_action_window_mask_rejects_invalid_window():
    with pytest.raises(ValueError, match="end"):
        action_window_mask(start=3, end=2, trajectory_length=4)
    with pytest.raises(ValueError, match="trajectory_length"):
        action_window_mask(start=0, end=2, trajectory_length=-1)


def test_all_one_mask_matches_unmasked_mse():
    prediction = torch.randn(3, 5, 2)
    target = torch.randn(3, 5, 2)
    mask = torch.ones(3, 5, dtype=torch.bool)

    assert torch.allclose(
        masked_mse_loss(prediction, target, mask), F.mse_loss(prediction, target)
    )


def test_all_zero_mask_is_finite_and_has_zero_gradient():
    prediction = torch.randn(2, 4, 3, requires_grad=True)
    target = torch.randn(2, 4, 3)
    mask = torch.zeros(2, 4, dtype=torch.bool)

    loss = masked_mse_loss(prediction, target, mask)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_padding_does_not_change_per_sample_normalisation():
    # Sample 0 has one valid element; sample 1 has all four.  The result is
    # the mean of the two per-sample errors, rather than a global element mean
    # that would over-weight the longer window.
    prediction = torch.tensor(
        [[[2.0], [0.0], [0.0], [0.0]], [[1.0], [3.0], [5.0], [7.0]]]
    )
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False, False, False], [True, True, True, True]])

    expected = torch.tensor((4.0 + (1.0 + 9.0 + 25.0 + 49.0) / 4.0) / 2.0)
    assert torch.allclose(masked_mse_loss(prediction, target, mask), expected)
