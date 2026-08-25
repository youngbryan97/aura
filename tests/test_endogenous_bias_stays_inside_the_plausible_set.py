"""The bias re-ranks; it cannot invent.

A half-trained head loose in a decode loop is how a language pathway produces
word salad. This one adds its bias only to tokens the model already finds
plausible this step, so the worst it can do is reorder near-ties. That
property is the safety argument, so it is tested rather than asserted.
"""

from __future__ import annotations

import numpy as np

from core.brain.llm.endogenous_decode import (
    DEFAULT_BETA,
    JOB_STATE_KEY,
    EndogenousLogitBiasProcessor,
    build_endogenous_processor,
    decision_is_expected_absence,
    observe_receipt,
    pathway_health,
    reset_head_cache,
    reset_pathway_health,
)
from core.brain.llm.endogenous_state import empty_state, layout_digest


def test_an_implausible_token_is_never_touched():
    processor = EndogenousLogitBiasProcessor(
        np.array([1.0, -1.0, 3.0, 0.0], dtype=np.float32)
    )
    logits = np.array([5.0, 4.9, -40.0, 0.1])
    out = processor.apply_numpy(logits)
    assert out[2] == logits[2], "a ruled-out token was promoted"
    assert out[0] == logits[0] + 1.0
    assert out[1] == logits[1] - 1.0


def test_a_flat_distribution_admits_every_token():
    processor = EndogenousLogitBiasProcessor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    out = processor.apply_numpy(np.zeros(3))
    assert np.allclose(out, np.array([1.0, 2.0, 3.0]))


def test_a_shape_mismatch_falls_open():
    processor = EndogenousLogitBiasProcessor(np.ones(4, dtype=np.float32))
    logits = np.array([1.0, 2.0, 3.0])
    assert np.allclose(processor.apply_numpy(logits), logits)


def test_non_finite_logits_fall_open():
    processor = EndogenousLogitBiasProcessor(np.ones(3, dtype=np.float32))
    logits = np.array([1.0, float("nan"), 3.0])
    out = processor.apply_numpy(logits)
    assert np.array_equal(out, logits, equal_nan=True)


def test_the_beta_floor_decides_the_admitted_set():
    delta = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    logits = np.array([0.0, -1.0, -6.0])
    wide = EndogenousLogitBiasProcessor(delta, beta=0.001).apply_numpy(logits)
    narrow = EndogenousLogitBiasProcessor(delta, beta=0.9).apply_numpy(logits)
    assert wide[2] != logits[2], "a permissive floor should admit the tail"
    assert narrow[1] == logits[1], "a strict floor should keep the tail out"
    assert DEFAULT_BETA > 0.0


class _Tokenizer:
    def get_vocab(self):
        return {f"tok{i}": i for i in range(32)}


def test_a_job_without_state_gets_a_receipt_not_a_processor():
    processor, receipt = build_endogenous_processor(_Tokenizer(), {})
    assert processor is None
    assert receipt["reason"] == "no_state_on_job"
    assert decision_is_expected_absence(receipt["reason"])


def test_a_job_with_state_and_no_head_says_so(tmp_path):
    reset_head_cache()
    state = empty_state().do(**{"uncertainty.confidence": 0.7})
    job = {JOB_STATE_KEY: state.to_payload()}
    processor, receipt = build_endogenous_processor(
        _Tokenizer(), job, directory=tmp_path / "absent"
    )
    assert processor is None
    assert receipt["reason"].startswith("no_head")
    assert decision_is_expected_absence(receipt["reason"])
    reset_head_cache()


def test_a_state_from_another_layout_is_refused(tmp_path):
    reset_head_cache()
    payload = empty_state().to_payload()
    payload["layout"] = "0" * 32
    processor, receipt = build_endogenous_processor(
        _Tokenizer(), {JOB_STATE_KEY: payload}, directory=tmp_path
    )
    assert processor is None
    assert receipt["reason"] == "state_payload_rejected"
    assert not decision_is_expected_absence(receipt["reason"])
    reset_head_cache()


def test_health_separates_expected_absence_from_a_fault():
    reset_pathway_health()
    observe_receipt({"reason": "no_head:no trained head on disk", "applied": False})
    observe_receipt({"reason": "tokenizer_mismatch", "applied": False})
    observe_receipt({"reason": "applied", "applied": True, "alpha": 0.6})
    health = pathway_health()
    assert health["generations_seen"] == 3
    assert health["bias_applied"] == 1
    assert health["unexpected_refusals"] == 1
    assert health["applied_share"] == round(1 / 3, 4)
    assert health["last_receipt"]["reason"] == "applied"
    reset_pathway_health()


def test_the_layout_travels_on_every_state_payload():
    assert empty_state().to_payload()["layout"] == layout_digest()


class TestTheFingerprintIsNotRecomputedPerGeneration:
    """Hashing a hundred thousand id-to-token pairs belongs outside the loop."""

    def test_the_same_tokenizer_is_hashed_once(self):
        from core.brain.llm.endogenous_decode import (
            cached_tokenizer_signature,
            reset_tokenizer_signature_cache,
        )

        calls: list[int] = []

        class _Counting:
            def get_vocab(self):
                calls.append(1)
                return {f"tok{i}": i for i in range(64)}

        reset_tokenizer_signature_cache()
        tokenizer = _Counting()
        first = cached_tokenizer_signature(tokenizer)
        second = cached_tokenizer_signature(tokenizer)
        assert first == second
        assert len(calls) == 1, "the vocabulary was walked twice for one tokenizer"
        reset_tokenizer_signature_cache()

    def test_a_different_tokenizer_is_hashed_again(self):
        from core.brain.llm.endogenous_decode import (
            cached_tokenizer_signature,
            reset_tokenizer_signature_cache,
        )

        class _Vocab:
            def __init__(self, offset: int) -> None:
                self._offset = offset

            def get_vocab(self):
                return {f"tok{i + self._offset}": i for i in range(64)}

        reset_tokenizer_signature_cache()
        assert cached_tokenizer_signature(_Vocab(0)) != cached_tokenizer_signature(
            _Vocab(1)
        )
        reset_tokenizer_signature_cache()


class TestTheWorkerMakesOneCall:
    """The worker owns degradation recording; the pathway owns the decision."""

    class _Tokenizer:
        def get_vocab(self):
            return {f"tok{i}": i for i in range(32)}

    def test_an_expected_absence_is_not_reported_as_a_fault(self):
        from core.brain.llm.endogenous_decode import install_endogenous_processor

        processors: list[object] = []
        receipt, fault = install_endogenous_processor(
            self._Tokenizer(), {}, processors
        )
        assert receipt["reason"] == "no_state_on_job"
        assert fault == ""
        assert processors == []

    def test_a_mismatch_is(self):
        from core.brain.llm.endogenous_decode import (
            JOB_STATE_KEY,
            install_endogenous_processor,
        )
        from core.brain.llm.endogenous_state import empty_state

        payload = empty_state().to_payload()
        payload["layout"] = "0" * 32
        _receipt, fault = install_endogenous_processor(
            self._Tokenizer(), {JOB_STATE_KEY: payload}, []
        )
        assert fault == "state_payload_rejected"
