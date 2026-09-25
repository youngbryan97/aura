"""A nested selector never trains on the proposer or selector holdout."""

import pytest

from tools.train_nested_semantic_ranker import validate_nested_provenance


def _plans():
    outer = {"fold": 0, "fit_ids": ["a", "b", "c", "d"],
             "calibration_ids": ["e", "f"], "held_ids": ["x", "y"],
             "outer_fold": None, "source_report_sha256": "source",
             "parent_receipt_sha256": "parent", "folds_sha256": "folds"}
    pairs = []
    for index, held in enumerate((("a", "b"), ("c", "d"), ("e", "f"))):
        plan = {"outer_fold": 0, "fold": index, "outer_excluded_ids": ["x", "y"],
                "source_report_sha256": "source", "parent_receipt_sha256": "parent",
                "folds_sha256": "folds", "inner_folds_receipt_sha256": "inner",
                "inner_fold_seed": 1,
                "fit_ids": [identity for identity in "abcdef" if identity not in held][:3],
                "calibration_ids": [identity for identity in "abcdef"
                                    if identity not in held][3:],
                "held_ids": list(held)}
        pairs.append((plan, {"row_receipts": dict.fromkeys(held, "receipt")}))
    return outer, {"row_receipts": {"x": "receipt", "y": "receipt"}}, pairs


def test_nested_provenance_covers_outer_training_without_leakage():
    outer, report, pairs = _plans()
    assert validate_nested_provenance(outer, report, pairs) == {
        "a": 0, "b": 0, "c": 1, "d": 1, "e": 2, "f": 2}


def test_nested_provenance_rejects_outer_held_in_any_inner_fit():
    outer, report, pairs = _plans()
    pairs[0][0]["fit_ids"][0] = "x"
    with pytest.raises(ValueError, match="holdout"):
        validate_nested_provenance(outer, report, pairs)


def test_nested_provenance_rejects_missing_training_bank():
    outer, report, pairs = _plans()
    pairs[2][0]["held_ids"].remove("f")
    with pytest.raises(ValueError, match="holdout"):
        validate_nested_provenance(outer, report, pairs)


def test_nested_provenance_rejects_mixed_inner_partitions():
    outer, report, pairs = _plans()
    pairs[2][0]["inner_folds_receipt_sha256"] = "different"
    with pytest.raises(ValueError, match="cover"):
        validate_nested_provenance(outer, report, pairs)


def test_nested_provenance_rejects_mixed_geometry_seed():
    outer, report, pairs = _plans()
    pairs[1][0]["inner_fold_seed"] = 2
    with pytest.raises(ValueError, match="cover"):
        validate_nested_provenance(outer, report, pairs)
