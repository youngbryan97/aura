"""G12 pairs her and her base model request by request."""

from __future__ import annotations

import json

from tools.g12_analysis import paired, rows, two_sided_mcnemar, wilson


def test_the_exact_two_sided_mcnemar_matches_its_definition() -> None:
    assert two_sided_mcnemar(0, 0) == 1.0
    assert two_sided_mcnemar(6, 0) == 0.03125
    assert two_sided_mcnemar(0, 6) == 0.03125
    assert two_sided_mcnemar(3, 3) == 1.0


def test_the_wilson_interval_brackets_the_rate() -> None:
    low, high = wilson(45, 50)
    assert low < 0.9 < high and 0 <= low and high <= 1


def test_requests_are_paired_by_key_and_missing_ones_counted(tmp_path) -> None:
    for system, results in (("her", {"a": True, "b": True, "c": False}), ("base", {"a": True, "b": False})):
        directory = tmp_path / system / "rows"
        directory.mkdir(parents=True)
        for key, correct in results.items():
            (directory / f"{key}.json").write_text(json.dumps({"unique_id": key, "correct": correct, "level": 1}))
    her, base = rows(tmp_path / "her", "unique_id"), rows(tmp_path / "base", "unique_id")
    result = paired(her, base, "correct", "level")
    assert result["requests"] == 2 and result["her"] == 2 and result["base"] == 1
    assert result["her_only"] == 1 and result["base_only"] == 0
    assert result["missing"] == {"her": 0, "base": 1}
    assert result["by_level"] == {"1": {"n": 2, "her": 2, "base": 1}}
