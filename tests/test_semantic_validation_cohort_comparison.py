"""Paired validation comparison cannot turn unpaired or corrupt rows into gains."""

import copy

import pytest

from core.learning.semantic_validation_checkpoint import _digest
from tools.compare_semantic_validation_cohorts import compare


def _signed(body, key):
    return {**body, key: _digest(body)}


def _fixtures():
    old = _signed({"schema": "aura.semantic_validation_checkpoint.v1", "identity": "old",
                   "rows": {"incumbent": {"a": [True, True], "b": [False, False],
                                          "c": [False, False], "d": [True, True]}}}, "sha256")
    statuses = {"a": "wrong", "b": "equivalent", "c": "wrong", "d": "equivalent"}
    rows = [_signed({"split": "validation", "source_text_sha256": source,
                     "observation": {"semantic_status": status},
                     "construction_id": "construction", "topology_id": "topology"},
                    "receipt_sha256") for source, status in statuses.items()]
    new = _signed({"schema": "aura.semantic_cohort_diagnosis.v2", "rows": rows},
                  "receipt_sha256")
    return old, new


def test_comparison_keeps_oracle_separate_from_real_candidate_scores():
    old, new = _fixtures()
    result = compare(old, new, old_name="incumbent")
    assert result["counts"] == {"both": 1, "old_only": 1, "new_only": 1, "neither": 1}
    assert (result["old_correct"], result["new_correct"], result["oracle_union"]) == (2, 2, 3)
    assert not result["protocol_matched"]
    assert not result["serving_authority"]
    assert not result["qualification_evidence"]


@pytest.mark.parametrize("change", ["unpaired", "row_tamper", "report_tamper"])
def test_comparison_rejects_unpaired_or_tampered_evidence(change):
    old, new = _fixtures()
    if change == "unpaired":
        new["rows"] = new["rows"][:-1]
        new = _signed({key: value for key, value in new.items()
                       if key != "receipt_sha256"}, "receipt_sha256")
    elif change == "row_tamper":
        new["rows"][0]["observation"]["semantic_status"] = "equivalent"
        new = _signed({key: value for key, value in new.items()
                       if key != "receipt_sha256"}, "receipt_sha256")
    else:
        new = copy.deepcopy(new)
        new["rows"][0]["observation"]["semantic_status"] = "equivalent"
    with pytest.raises(ValueError):
        compare(old, new, old_name="incumbent")
