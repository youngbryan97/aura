"""The paired trial of her one-shot store: invented facts no weights hold, and an exact test on what disagrees."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "nonparametric_proof.py"
_spec = importlib.util.spec_from_file_location("nonparametric_proof", _TOOL)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_the_facts_are_invented_varied_and_reproducible():
    first = tool.invented_facts(40, seed=7)
    assert first == tool.invented_facts(40, seed=7)
    assert len({prompt for prompt, _ in first}) == 40
    kinds = ("torque limit is", "the codename", "drawdown rate is", "call sign is", "rated for a load of", "on shelf")
    assert all(any(prompt.endswith(kind) for prompt, _ in first) for kind in kinds)
    assert first != tool.invented_facts(40, seed=8)


def test_the_trial_holds_only_when_the_store_alone_answers_and_no_control_moves():
    won = [{"memory_has_it": True, "base_had_it": False}] * 12
    held = tool._paired(won, controls_changed=0)
    assert held["store_only"] == 12 and held["model_only"] == 0
    assert held["p_one_sided"] == 0.5 ** 12 and held["holds"]
    assert not tool._paired(won, controls_changed=1)["holds"]
    split = won[:5] + [{"memory_has_it": False, "base_had_it": True}] * 5
    assert not tool._paired(split, controls_changed=0)["holds"]
