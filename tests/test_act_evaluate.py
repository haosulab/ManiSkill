import importlib.util
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch


def load_act_evaluate():
    common = types.ModuleType("common")
    utils = types.ModuleType("mani_skill.utils")
    utils.common = common

    sys.modules["mani_skill"] = types.ModuleType("mani_skill")
    sys.modules["mani_skill.utils"] = utils
    sys.modules["mani_skill.utils.common"] = common

    evaluate_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "baselines"
        / "act"
        / "act"
        / "evaluate.py"
    )
    spec = importlib.util.spec_from_file_location("act_evaluate", evaluate_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_append_eval_metrics_reads_vector_final_info_dict():
    module = load_act_evaluate()
    metrics = defaultdict(list)

    module._append_eval_metrics(
        metrics,
        {
            "final_info": {
                "episode": {
                    "return": torch.tensor([1.0, 2.0]),
                    "success_at_end": torch.tensor([True, False]),
                }
            }
        },
    )

    np.testing.assert_array_equal(metrics["return"][0], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(metrics["success_at_end"][0], np.array([1.0, 0.0]))


def test_append_eval_metrics_reads_vector_final_info_list():
    module = load_act_evaluate()
    metrics = defaultdict(list)

    module._append_eval_metrics(
        metrics,
        {
            "final_info": [
                {"episode": {"return": np.array(1.0)}},
                {"episode": {"return": np.array(2.0)}},
            ]
        },
    )

    assert metrics["return"] == [np.array(1.0), np.array(2.0)]


def test_append_eval_metrics_falls_back_to_cpu_episode_info():
    module = load_act_evaluate()
    metrics = defaultdict(list)

    module._append_eval_metrics(
        metrics,
        {
            "episode": {
                "return": np.array([1.0, 2.0]),
                "_return": np.array([True, True]),
                "success_at_end": np.array([True, False]),
                "_success_at_end": np.array([True, True]),
            }
        },
    )

    assert set(metrics) == {"return", "success_at_end"}
    np.testing.assert_array_equal(metrics["return"][0], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(metrics["success_at_end"][0], np.array([True, False]))


def test_append_eval_metrics_requires_episode_metrics():
    module = load_act_evaluate()

    with pytest.raises(KeyError, match="final_info.*episode"):
        module._append_eval_metrics(defaultdict(list), {})
