"""G05's plan takes its effect from the closed-channel pilot's rows, and refuses rows decoded otherwise."""

from __future__ import annotations

import json

import pytest

from tools.g05_public_answer_protocol import build_strata, pilot_effect


def _write(root, stratum, arm, task, exact, thinking="closed"):
    path = root / stratum / "rows" / arm / f"{task}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task_id": task, "answer_exact": exact, "thinking": thinking}))


def test_the_effect_is_bounded_from_paired_rows(tmp_path) -> None:
    for index in range(24):
        _write(tmp_path, "composition", "assisted", f"t{index}", True)
        _write(tmp_path, "composition", "ordinary", f"t{index}", index >= 8)
    _write(tmp_path, "composition_v2", "assisted", "u0", True)
    _write(tmp_path, "composition_v2", "ordinary", "u0", True)
    effect = pilot_effect(tmp_path, "composition")
    # Every directory named for the stratum counts: 24 pairs here and one in composition_v2.
    assert effect["assisted_only"] == 8 and effect["ordinary_only"] == 0 and effect["pilot_pairs"] == 25
    # Bounds, not the point values 8/24 and 8/8.
    assert 0 < effect["discordance"] < 8 / 25 and 0.5 < effect["win_share"] < 1.0


def test_rows_decoded_with_the_channel_open_are_refused(tmp_path) -> None:
    _write(tmp_path, "lists", "assisted", "t0", True, thinking="open")
    _write(tmp_path, "lists", "ordinary", "t0", False)
    with pytest.raises(SystemExit, match="closed"):
        pilot_effect(tmp_path, "lists")


def test_a_pilot_without_a_gain_plans_nothing(tmp_path) -> None:
    for index in range(4):
        _write(tmp_path, "lists", "assisted", f"t{index}", True)
        _write(tmp_path, "lists", "ordinary", f"t{index}", True)
    with pytest.raises(SystemExit, match="no paired gain"):
        pilot_effect(tmp_path, "lists")


def test_the_strata_come_from_the_generators_in_whole_cells() -> None:
    strata, per_cell = build_strata(11, {"composition": 30, "lists": 5})
    assert per_cell == 2 and len(strata["composition"]) == 48 and len(strata["lists"]) == 5


def test_too_few_disagreements_to_plan_for_are_refused(tmp_path) -> None:
    for index in range(24):
        _write(tmp_path, "lists", "assisted", f"t{index}", True)
        _write(tmp_path, "lists", "ordinary", f"t{index}", index >= 3)
    with pytest.raises(SystemExit, match="too few disagreements"):
        pilot_effect(tmp_path, "lists")


def test_the_sham_comparison_counts_the_other_way(tmp_path) -> None:
    for index in range(24):
        _write(tmp_path, "lists", "ordinary", f"t{index}", True)
        _write(tmp_path, "lists", "sham", f"t{index}", index >= 12)
    effect = pilot_effect(tmp_path, "lists", first="ordinary", second="sham")
    assert effect["ordinary_only"] == 12 and effect["sham_only"] == 0
