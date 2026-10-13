"""Unit tests for PDEEPosController IK-failure observability.

These stub out the sim-dependent pieces so they do not need a GPU or a
render backend. See https://github.com/haosulab/ManiSkill/issues/1482.
"""
from types import SimpleNamespace

import torch
from gymnasium import spaces

from mani_skill.agents.controllers import pd_ee_pose
from mani_skill.agents.controllers.pd_ee_pose import (
    PDEEPosController,
    PDEEPosControllerConfig,
)


class _StubPDEEPosController(PDEEPosController):
    """Keep the real set_action/reset path; stub out sim-dependent pieces."""

    @property
    def ee_pose_at_base(self):
        return None

    def compute_target_pose(self, prev_ee_pose_at_base, action):
        return None

    def set_drive_targets(self, targets):
        self._drive_targets = targets


def _make_controller(ik_result=None, **config_kwargs):
    """Build a sim-free controller whose compute_ik returns ik_result (None = failure)."""
    config = PDEEPosControllerConfig(
        joint_names=[],
        stiffness=0,
        damping=0,
        ee_link="ee",
        urdf_path="",
        pos_lower=-0.1,
        pos_upper=0.1,
        **config_kwargs,
    )
    controller = _StubPDEEPosController.__new__(_StubPDEEPosController)
    controller.config = config
    controller.scene = SimpleNamespace(
        num_envs=1, gpu_sim_enabled=False, _reset_mask=torch.tensor([True])
    )
    controller.active_joint_indices = torch.tensor([0, 1])
    controller.articulation = SimpleNamespace(
        get_qpos=lambda: torch.zeros((1, 2)),
        device=torch.device("cpu"),
    )
    controller.kinematics = SimpleNamespace(compute_ik=lambda **kwargs: ik_result)
    controller._normalize_action = False
    controller.single_action_space = spaces.Box(
        low=-0.1, high=0.1, shape=(3,), dtype="float32"
    )
    controller.action_space = controller.single_action_space
    controller._target_qpos = torch.zeros((1, 2))
    controller._start_qpos = torch.zeros((1, 2))
    controller.ik_failure_count = 0
    controller._target_pose = None
    return controller


def test_ik_failure_holds_targets_and_counts():
    controller = _make_controller(ik_result=None)
    action = torch.zeros((1, 3))
    controller.set_action(action)
    assert torch.equal(controller._target_qpos, controller._start_qpos)
    assert controller.ik_failure_count == 1
    controller.set_action(action)
    assert controller.ik_failure_count == 2


def test_ik_success_does_not_count():
    target = torch.ones((1, 2))
    controller = _make_controller(ik_result=target)
    controller.set_action(torch.zeros((1, 3)))
    assert torch.equal(controller._target_qpos, target)
    assert controller.ik_failure_count == 0


def test_ik_failure_count_reset_on_reset():
    controller = _make_controller(ik_result=None)
    controller.set_action(torch.zeros((1, 3)))
    assert controller.ik_failure_count == 1
    controller.reset()
    assert controller.ik_failure_count == 0


def test_no_warning_by_default(monkeypatch):
    calls = []
    monkeypatch.setattr(pd_ee_pose.logger, "warning", lambda msg: calls.append(msg))
    controller = _make_controller(ik_result=None)
    assert controller.config.warn_on_ik_failure is False
    controller.set_action(torch.zeros((1, 3)))
    assert controller.ik_failure_count == 1
    assert calls == []


def test_warn_on_ik_failure_is_rate_limited(monkeypatch):
    calls = []
    monkeypatch.setattr(pd_ee_pose.logger, "warning", lambda msg: calls.append(msg))
    controller = _make_controller(ik_result=None, warn_on_ik_failure=True)
    for _ in range(101):
        controller.set_action(torch.zeros((1, 3)))
    assert controller.ik_failure_count == 101
    # first failure, then every 100th
    assert len(calls) == 2
    assert "holding previous joint targets" in calls[0]
