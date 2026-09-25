"""A frozen selector can be compared only on its own held construction fold."""

from tools.evaluate_nested_semantic_ranker_variant import (
    _digest,
    choice_overlap,
    verify_variant,
)


def _fixtures():
    outer = {"fold": 0, "outer_fold": None, "source_report_sha256": "source",
             "parent_receipt_sha256": "parent", "folds_sha256": "folds",
             "fit_ids": ["fit"], "calibration_ids": ["cal"], "held_ids": ["held"]}
    variant = {**outer, "held_ids": ["held"]}
    report = {"row_receipts": {"held": "receipt"}}
    body = {"schema": "aura.semantic_nested_ranker_source_fold.v1",
            "weights_sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "outer_fold": 0, "source_report_sha256": "source"}
    training = {**body, "receipt_sha256": _digest(body)}
    return training, outer, variant, report


def test_variant_requires_matching_source_and_partitions():
    training, outer, variant, report = _fixtures()
    assert verify_variant(training, outer, variant, report, b"abc") == ("held",)
    for changed in ({**variant, "source_report_sha256": "other"},
                    {**variant, "fit_ids": ["held"]},
                    {**variant, "held_ids": ["foreign"]}):
        try:
            verify_variant(training, outer, changed, report, b"abc")
        except ValueError:
            pass
        else:
            raise AssertionError("cross-fold variant was accepted")


def test_choice_overlap_keeps_joint_and_ranking_outcomes_separate():
    rows = {"one": {"bank": {"candidates": [
        {"program_sha256": "a", "joint_score": 0.1},
        {"program_sha256": "b", "joint_score": 0.9}]},
        "diagnosis": {"comparisons": [
            {"program_sha256": "a", "status": "different"},
            {"program_sha256": "b", "status": "equivalent"}]}}}
    evaluation = {"population": 1, "incumbent_correct": 0, "ranker_correct": 0,
                  "rows": [{"source": "one", "incumbent_correct": False,
                            "selected_correct": False}]}
    assert choice_overlap(rows, evaluation) == {
        "ordinary_correct": 0, "joint_correct": 1, "ranker_correct": 0,
        "oracle_union": 1, "joint_patterns": {"0-1-0": 1}}
