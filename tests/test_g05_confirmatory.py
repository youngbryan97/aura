"""G05's analysis follows the plan from the rows alone."""

from __future__ import annotations

import json

from tools.g05_confirmatory import analyse

SPEC = {
    "domains": {"composition": [{"id": f"c{i}"} for i in range(14)]},
    "secondary_domains": {"lists": [{"id": f"l{i}"} for i in range(14)]},
    "arms_by_stratum": {"composition": ["ordinary", "assisted", "sham", "ordinary_open"],
                        "lists": ["ordinary", "assisted", "sham"]},
    "secondary": {"per_test_alpha": 0.025, "comparisons": [
        {"first": "assisted", "second": "ordinary_open", "stratum": "composition", "alternative": "two-sided"},
        {"first": "ordinary", "second": "sham", "stratum": "lists", "alternative": "greater"},
    ]},
}
PLAN = {"parameters": {"per_comparison_alpha": 0.05}}


def _write(run, stratum, arm, task, exact, reader=None):
    path = run / stratum / "rows" / arm / f"{task}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task_id": task, "answer_exact": exact, "expected": 7,
                                "reader_answer": reader, "generated_tokens": 10, "seconds": 1.0,
                                "termination": "stop", "thinking": "closed"}))


def test_the_closure_rule_needs_the_primary_rejection_and_every_translation(tmp_path) -> None:
    for i in range(14):
        _write(tmp_path, "composition", "ordinary", f"c{i}", i >= 10)
        _write(tmp_path, "composition", "assisted", f"c{i}", True, reader=7)
        _write(tmp_path, "composition", "sham", f"c{i}", True)
        _write(tmp_path, "composition", "ordinary_open", f"c{i}", True)
        _write(tmp_path, "lists", "ordinary", f"l{i}", True)
        _write(tmp_path, "lists", "assisted", f"l{i}", True, reader=7)
        _write(tmp_path, "lists", "sham", f"l{i}", i >= 10)
    report = analyse(SPEC, PLAN, tmp_path)
    assert report["complete"] and report["primary"]["assisted_only"] == 10 and report["primary"]["rejects"]
    assert report["translation"] == {"reader_right": 28, "assisted_exact_where_reader_right": 28, "holds": True}
    assert report["secondary"]["ordinary_vs_sham:lists"]["first_only"] == 10
    assert report["g05_closure_rule_holds"]

    _write(tmp_path, "lists", "assisted", "l0", False, reader=7)
    report = analyse(SPEC, PLAN, tmp_path)
    assert not report["translation"]["holds"] and not report["g05_closure_rule_holds"]


def test_a_missing_row_rejects_nothing(tmp_path) -> None:
    for i in range(13):
        _write(tmp_path, "composition", "ordinary", f"c{i}", False)
        _write(tmp_path, "composition", "assisted", f"c{i}", True, reader=7)
    report = analyse(SPEC, PLAN, tmp_path)
    assert not report["complete"] and not report["primary"]["rejects"]
