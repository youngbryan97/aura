"""The paired triadic probe compares the same sources and outcomes."""

from types import SimpleNamespace

import pytest

from tools.probe_semantic_triadic_pair import graph_refit_calibration_sources, paired_accuracy


def _arm(answer: tuple[bool, ...], program: tuple[bool, ...]) -> dict:
    return {"rows": [
        {"source_text_sha256": str(index), "answer_exact": answer[index],
         "program_exact": program[index], "argument_exact": program[index]}
        for index in range(len(answer))]}


def test_paired_accuracy_counts_gains_regressions_and_lesion():
    incumbent = _arm((True, False, True), (True, False, False))
    candidate = _arm((True, True, False), (True, True, False))
    lesion = _arm((True, False, False), (True, False, False))
    result = paired_accuracy(incumbent, candidate, lesion)
    assert result["answer_exact"] == {"incumbent": 2, "candidate": 2, "lesion": 1,
                                      "candidate_only": 1, "incumbent_only": 1,
                                      "candidate_and_lesion": 1}
    assert result["program_exact"]["candidate_only"] == 1


def test_paired_accuracy_refuses_misaligned_rows():
    left = _arm((True, False), (True, False))
    right = _arm((False, True), (False, True))
    right["rows"].reverse()
    with pytest.raises(ValueError, match="source identity or order"):
        paired_accuracy(left, right, left)


def test_uncalibrated_graph_refit_replays_the_fold_cohort():
    candidate = SimpleNamespace(training_receipt={"triadic_binding_fit": {"score_calibration": None}})
    source_validation = SimpleNamespace(split="validation")
    fold_calibration = SimpleNamespace(split="train")
    assert graph_refit_calibration_sources(
        candidate, (source_validation, fold_calibration), (fold_calibration,)
    ) == (fold_calibration,)
