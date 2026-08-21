"""The measurement that decides whether the learned substrate is real.

The claim under test is narrow: reading a sentence off the resident model's
hidden states separates "this reply claims a completed action" from
near-misses better than a topical embedder does, on wordings the boundary was
never fitted to.

These tests cover the protocol and the arithmetic. The answer itself is
produced against the live model and filed at
artifacts/language_substrate/measurement.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.language.substrate_measurement import (
    load_frozen_set,
    measure_separation,
    roc_auc,
)

ROOT = Path(__file__).resolve().parents[1]


def test_the_area_is_right_at_the_edges() -> None:
    assert roc_auc([0.1, 0.2, 0.9, 1.0], [False, False, True, True]) == 1.0
    assert roc_auc([0.9, 1.0, 0.1, 0.2], [False, False, True, True]) == 0.0
    assert roc_auc([0.5] * 4, [False, False, True, True]) == 0.5


def test_one_class_has_no_area() -> None:
    """0.5 would report a coin flip that was never tossed."""
    assert roc_auc([0.1, 0.2], [True, True]) is None
    assert roc_auc([], []) is None


def test_the_frozen_set_is_held_out() -> None:
    """Nothing evaluated may also be an example."""
    from core.conversation.response_reliability import _ACTION_CLAIM_MATCHER

    held_out, digest = load_frozen_set()
    assert len(held_out) >= 20
    assert digest
    declared = set(_ACTION_CLAIM_MATCHER.positives) | set(_ACTION_CLAIM_MATCHER.negatives)
    assert not {sentence for sentence, _label in held_out} & declared


def test_the_frozen_set_has_both_classes_and_near_misses() -> None:
    held_out, _digest = load_frozen_set()
    positives = sum(1 for _s, label in held_out if label)
    assert positives >= 8
    assert len(held_out) - positives >= 8


def test_a_separable_space_scores_perfectly() -> None:
    def mood(sentences):
        return [
            [1.0 if str(s).lower().startswith("i ") else 0.0, 1.0 if "?" in s else 0.0]
            for s in sentences
        ]

    measurement = measure_separation(
        feature_source=mood,
        source_name="mood",
        positives=("I did it.", "I closed it.", "I wrote it."),
        negatives=("Would you do it?", "Shall I close it?", "Could you write it?"),
        held_out=[("I moved the file.", True), ("Would you move the file?", False)],
    )
    assert measurement.auroc == 1.0
    assert measurement.abstain_rate == 0.0
    assert measurement.false_positive_rate == 0.0


def test_abstentions_are_reported_as_themselves() -> None:
    """A measurement that scored them as errors, or dropped them, would
    describe a system nobody runs."""
    measurement = measure_separation(
        feature_source=lambda sentences: [[0.5] for _ in sentences],
        source_name="flat",
        positives=("a", "b", "c"),
        negatives=("d", "e", "f"),
        held_out=[("x", True), ("y", False)],
    )
    assert measurement.abstain_rate == 1.0
    assert measurement.decided == 0
    assert measurement.f1 is None


def test_a_measurement_never_touches_what_production_learned(tmp_path) -> None:
    """It must not read or write the live matcher's store."""
    import inspect

    from core.language import substrate_measurement

    source = inspect.getsource(substrate_measurement.measure_separation)
    assert "matcher._loaded = True" in source


def test_the_receipt_records_both_feature_spaces_when_present() -> None:
    receipt = ROOT / "artifacts" / "language_substrate" / "measurement.json"
    if not receipt.is_file():
        return
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload = payload.get("payload", payload)
    sources = {row["feature_source"] for row in payload.get("results", [])}
    assert {"topical_embedding", "model_hidden_state"} <= sources
