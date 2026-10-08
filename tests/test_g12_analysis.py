"""G12 pairs her and her base model request by request."""

from __future__ import annotations

import json

import pytest

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
    result = paired(her, base, "correct", "level", expected_keys=["a", "b", "c"])
    assert result["requests"] == 2 and result["her"] == 2 and result["base"] == 1
    assert result["her_only"] == 1 and result["base_only"] == 0
    assert result["missing"] == {"her": 0, "base": 1}
    assert result["by_level"] == {"1": {"n": 2, "her": 2, "base": 1}}


def test_matching_incomplete_arms_cannot_claim_source_coverage() -> None:
    same = {"a": {"correct": True}}
    unknown = paired(same, same, "correct")
    assert unknown["unpaired"] == {"her": 0, "base": 0}
    assert unknown["missing"] is None and unknown["coverage"]["complete"] is None
    known = paired(same, same, "correct", expected_keys=["a", "omitted"])
    assert known["missing"] == {"her": 1, "base": 1}
    assert known["coverage"] == {"expected": 2, "her": 1, "base": 1, "complete": False}


def test_duplicate_saved_ids_are_rejected_instead_of_last_wins(tmp_path) -> None:
    directory = tmp_path / "rows"
    directory.mkdir()
    for name, correct in (("first", True), ("last", False)):
        (directory / f"{name}.json").write_text(json.dumps({"id": "same", "correct": correct}))
    with pytest.raises(ValueError, match="duplicate result case ID"):
        rows(tmp_path, "id")


def test_source_expectation_rejects_duplicates_and_foreign_results() -> None:
    with pytest.raises(ValueError, match="not unique"):
        paired({}, {}, "correct", expected_keys=["a", "a"])
    with pytest.raises(ValueError, match="unexpected source case"):
        paired({"foreign": {"correct": True}}, {}, "correct", expected_keys=["a"])


def test_empty_source_expectation_does_not_certify_an_unmeasured_run() -> None:
    result = paired({}, {}, "correct", expected_keys=[])
    assert result["requests"] == 0 and result["coverage"]["expected"] == 0
    assert result["coverage"]["complete"] is None
