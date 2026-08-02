"""Unit tests for PDEEPoseController action scaling.

These exercise ``_clip_and_scale_action`` directly so they do not need a GPU or a
render backend. See https://github.com/haosulab/ManiSkill/issues/1469 for the
rotation sign bug this guards against.
"""
import pytest
import torch

from mani_skill.agents.controllers.pd_ee_pose import (
    PDEEPoseController,
    PDEEPoseControllerConfig,
)


def _make_scaler(rot_lower, rot_upper, pos_lower=-0.1, pos_upper=0.1):
    """A stand-in exposing only what ``_clip_and_scale_action`` reads."""
    config = PDEEPoseControllerConfig(
        joint_names=[],
        stiffness=0,
        damping=0,
        ee_link="ee",
        urdf_path="",
        pos_lower=pos_lower,
        pos_upper=pos_upper,
        rot_lower=rot_lower,
        rot_upper=rot_upper,
        frame="root_translation:root_aligned_body_rotation",
    )
    scaler = PDEEPoseController.__new__(PDEEPoseController)
    scaler.config = config
    scaler.action_space_low = torch.tensor(
        [pos_lower] * 3 + [rot_lower] * 3, dtype=torch.float32
    )
    scaler.action_space_high = torch.tensor(
        [pos_upper] * 3 + [rot_upper] * 3, dtype=torch.float32
    )
    return scaler


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_rotation_action_preserves_sign(axis):
    scaler = _make_scaler(rot_lower=-0.5, rot_upper=0.5)
    action = torch.zeros((1, 6), dtype=torch.float32)
    action[0, 3 + axis] = 1.0
    scaled = scaler._clip_and_scale_action(action)
    # a positive normalized rotation about an axis must stay positive, matching
    # the translation semantics
    assert scaled[0, 3 + axis].item() > 0
    action[0, 3 + axis] = -1.0
    scaled = scaler._clip_and_scale_action(action)
    assert scaled[0, 3 + axis].item() < 0


def test_rotation_action_scaled_to_bound():
    scaler = _make_scaler(rot_lower=-0.5, rot_upper=0.5)
    action = torch.zeros((1, 6), dtype=torch.float32)
    action[0, 5] = 1.0
    scaled = scaler._clip_and_scale_action(action)
    assert scaled[0, 5].item() == pytest.approx(0.5)


def test_translation_and_rotation_agree_in_direction():
    scaler = _make_scaler(rot_lower=-0.2, rot_upper=0.2, pos_lower=-0.2, pos_upper=0.2)
    action = torch.ones((1, 6), dtype=torch.float32)
    scaled = scaler._clip_and_scale_action(action)
    # positive translation and positive rotation should both come out positive
    assert torch.all(scaled[0, :3] > 0)
    assert torch.all(scaled[0, 3:] > 0)


if __name__ == "__main__":
    for ax in range(3):
        test_rotation_action_preserves_sign(ax)
    test_rotation_action_scaled_to_bound()
    test_translation_and_rotation_agree_in_direction()
