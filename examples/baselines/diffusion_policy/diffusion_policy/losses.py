"""Losses used by the diffusion-policy baselines.

The action windows used by the ManiSkill demonstrations are fixed length.  A
window near an episode boundary therefore contains repeated/zero actions that
are only padding and are not expert demonstrations.  ``masked_mse_loss``
keeps those values out of the denoising objective while normalising every
sample by its own number of valid action elements.
"""

from typing import Optional

import torch


def action_window_mask(
    start: int,
    end: int,
    trajectory_length: int,
    device=None,
) -> torch.Tensor:
    """Return valid-demo positions for a half-open action window.

    ``start``/``end`` use trajectory coordinates and may extend beyond either
    episode boundary.  The returned vector always has ``end - start`` entries
    and is true exactly where an action index in ``[0, trajectory_length)`` is
    represented.  Keeping this calculation in one place makes state and RGB-D
    datasets agree on padding semantics.
    """

    if end < start:
        raise ValueError(f"window end ({end}) must be >= start ({start})")
    if trajectory_length < 0:
        raise ValueError("trajectory_length must be non-negative")

    mask = torch.zeros(end - start, dtype=torch.bool, device=device)
    valid_start = max(start, 0)
    valid_end = min(end, trajectory_length)
    if valid_end > valid_start:
        mask[valid_start - start : valid_end - start] = True
    return mask


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute a per-sample-normalised mean squared error.

    Args:
        prediction: Model output, usually ``(batch, horizon, action_dim)``.
        target: Tensor with the same shape as ``prediction``.
        valid_mask: Optional boolean/weight mask.  A ``(batch, horizon)``
            mask is expanded over the action dimension; a mask that is
            broadcastable to ``prediction`` is also accepted.  Masked values
            do not contribute to either the numerator or denominator.

    Returns:
        A scalar tensor.  Each batch element is averaged over its valid
        elements before the batch mean is taken.  An all-zero mask returns a
        finite zero loss and produces no gradient.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.ndim == 0:
        raise ValueError("prediction and target must include a batch dimension")

    squared_error = (prediction - target).square()
    if valid_mask is None:
        return squared_error.mean()

    if valid_mask.ndim == prediction.ndim - 1:
        # The common action-window form is (B, T), with one weight per action
        # timestep.  Only append this dimension when the leading dimensions
        # agree; otherwise the broadcast error below gives a useful message.
        valid_mask = valid_mask.unsqueeze(-1)
    if valid_mask.ndim != prediction.ndim:
        raise ValueError(
            "valid_mask must have one fewer or the same number of dimensions "
            f"as prediction, got {valid_mask.ndim} for {prediction.ndim}"
        )
    if valid_mask.shape[0] != prediction.shape[0]:
        raise ValueError(
            "valid_mask and prediction must have the same batch size, got "
            f"{valid_mask.shape[0]} and {prediction.shape[0]}"
        )

    try:
        expanded_mask = valid_mask.to(
            device=squared_error.device, dtype=squared_error.dtype
        ).expand_as(squared_error)
    except RuntimeError as exc:
        raise ValueError(
            "valid_mask must be broadcastable to prediction; got "
            f"{tuple(valid_mask.shape)} and {tuple(prediction.shape)}"
        ) from exc

    # Reduce each sample independently.  This prevents boundary windows with
    # fewer demonstrations from being down-weighted merely because they have
    # more padding than interior windows.
    flat_error = squared_error.reshape(squared_error.shape[0], -1)
    flat_mask = expanded_mask.reshape(expanded_mask.shape[0], -1)
    denominator = flat_mask.sum(dim=1).clamp_min(1)
    per_sample_loss = (flat_error * flat_mask).sum(dim=1) / denominator
    return per_sample_loss.mean()
