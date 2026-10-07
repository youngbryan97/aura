"""G04's second protocol: the analysis follows the plan, and each lesion removes exactly one fitted part."""

from __future__ import annotations

from types import SimpleNamespace

from core.learning.semantic_argument_antecedent import FEATURES, RECENCY_FEATURES, ArgumentAntecedent
from core.learning.semantic_operation_peaks import PeakRecognitionTransducer
from tests.test_semantic_operation_peaks import _head, _recognizer
from tools.g04_transfer_protocol_v2 import lesions
from tools.run_g04_transfer_v2 import analyse

ARMS = ["incumbent", "candidate", "v12", "no_phrase_reading", "no_recency"]


def _row(**right: bool) -> dict:
    return {arm: {"equivalent": right.get(arm, False), "answer_correct": right.get(arm, False),
                  "decoded_without_execution": True} for arm in ARMS}


def _spec() -> dict:
    return {
        "arms": ARMS,
        "domains": {"vocabulary": [], "depth": [], "family": []},
        "secondary": {"per_test_alpha": 0.025, "comparisons": [
            {"treatment": "candidate", "control": "no_phrase_reading", "stratum": "vocabulary"},
            {"treatment": "candidate", "control": "no_recency", "stratum": "depth"},
        ]},
    }


def test_the_closure_rule_needs_every_primary_rejection_and_all_of_construction() -> None:
    win = _row(candidate=True)
    rows = {name: {f"{name}{i}": win for i in range(12)} for name in ("vocabulary", "depth", "family")}
    rows["construction"] = {f"c{i}": _row(candidate=True, incumbent=True) for i in range(5)}
    report = analyse(_spec(), {"parameters": {"per_comparison_alpha": 0.05 / 3}}, rows)
    assert report["all_primary_reject"] and report["construction"]["holds"]
    assert report["g04_closure_rule_holds"]
    assert report["secondary"]["candidate_vs_no_phrase_reading:vocabulary"]["treatment_only"] == 12

    rows["construction"]["c0"] = _row(incumbent=True)
    report = analyse(_spec(), {"parameters": {"per_comparison_alpha": 0.05 / 3}}, rows)
    assert report["all_primary_reject"] and not report["construction"]["holds"]
    assert not report["g04_closure_rule_holds"]


def test_a_stratum_without_enough_wins_does_not_reject() -> None:
    rows = {name: {f"{name}{i}": _row(candidate=True) for i in range(12)} for name in ("vocabulary", "family")}
    rows["depth"] = {"d0": _row(candidate=True), "d1": _row(incumbent=True)}
    rows["construction"] = {"c0": _row(candidate=True)}
    report = analyse(_spec(), {"parameters": {"per_comparison_alpha": 0.05 / 3}}, rows)
    assert not report["primary"]["depth"]["rejects"] and not report["g04_closure_rule_holds"]


def test_each_lesion_takes_out_one_fitted_part_and_keeps_the_rest() -> None:
    names = ("add", "sub")
    reading = _head(names, [[0, 0, 4], [0, 0, -4]])
    recognizer = _recognizer(close_labeler=reading, phrase_labeler=reading, phrase_weights=(0.5, 1.5),
                             sentence_ends=frozenset({9}), punctuation=frozenset({9}))
    antecedent = ArgumentAntecedent(tuple(float(i + 1) for i in range(len(FEATURES) + len(RECENCY_FEATURES))),
                                    0.0, {}, recency=True)
    base = SimpleNamespace(receipt_sha256="b" * 64)
    candidate = PeakRecognitionTransducer(base, recognizer, None, antecedent)
    arms = lesions(candidate)
    phrase, recency = arms["no_phrase_reading"], arms["no_recency"]
    assert phrase.recognizer.close_labeler is None and phrase.recognizer.phrase_weights == (0.0, 0.0)
    assert phrase.antecedent is antecedent and phrase.base is base
    assert recency.recognizer is recognizer
    assert recency.antecedent.weight[-2:] == (0.0, 0.0)
    assert recency.antecedent.weight[:-2] == antecedent.weight[:-2]
    assert len({candidate.receipt_sha256, phrase.receipt_sha256, recency.receipt_sha256}) == 3
