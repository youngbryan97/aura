"""Every way the vocabulary head declines, and the reason it gives.

A head is bound to one channel layout and one tokenizer. Both are fingerprints
in the artifact, because a head that loads against the wrong model would
produce numbers, bias the wrong words, and carry a report saying it was
trained.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.brain.llm.endogenous_state import (
    STATE_DIM,
    empty_state,
    layout_digest,
    semantics_digest,
)
from core.brain.llm.endogenous_vocab_head import (
    MAX_ABS_BIAS,
    MIN_COVERAGE,
    EndogenousVocabHead,
    HeadUnusableError,
    alpha_from_env,
    tokenizer_signature,
    untrained_head,
)


def _trained(vocab: int = 256, tokenizer: str = "sig", seed: int = 0) -> EndogenousVocabHead:
    rng = np.random.default_rng(seed)
    return EndogenousVocabHead(
        weights=rng.standard_normal((vocab, STATE_DIM)).astype(np.float32) * 0.2,
        bias=np.zeros(vocab, dtype=np.float32),
        vocab_size=vocab,
        layout=layout_digest(),
        semantics=semantics_digest(),
        tokenizer=tokenizer,
        trained=True,
    )


def _covered_state():
    return empty_state().do(
        **{f"substrate.band_{i:02d}": 0.3 for i in range(16)}
    ).do(**{"uncertainty.confidence": 0.7})


def test_an_untrained_head_never_produces_a_bias():
    head = untrained_head(64, "sig")
    delta, decision = head.decide(_covered_state(), tokenizer_sig="sig", alpha=1.0)
    assert delta is None
    assert decision.reason == "head_untrained"
    assert decision.applied is False


def test_a_tokenizer_mismatch_refuses():
    _, decision = _trained().decide(_covered_state(), tokenizer_sig="other", alpha=1.0)
    assert decision.reason == "tokenizer_mismatch"


def test_a_layout_mismatch_refuses():
    head = _trained()
    moved = EndogenousVocabHead(
        weights=head.weights,
        bias=head.bias,
        vocab_size=head.vocab_size,
        layout="0" * 32,
        tokenizer="sig",
        trained=True,
    )
    _, decision = moved.decide(_covered_state(), tokenizer_sig="sig", alpha=1.0)
    assert decision.reason == "layout_mismatch"


def test_alpha_zero_disables_the_pathway():
    _, decision = _trained().decide(_covered_state(), tokenizer_sig="sig", alpha=0.0)
    assert decision.reason == "alpha_disabled"


def test_a_state_nothing_answered_for_produces_no_bias():
    _, decision = _trained().decide(empty_state(), tokenizer_sig="sig", alpha=1.0)
    assert decision.reason == "state_coverage_below_floor"
    assert decision.coverage < MIN_COVERAGE


def test_the_bias_is_centred_and_clipped():
    head = _trained(seed=3)
    delta = head.delta_logits(_covered_state())
    assert abs(float(np.mean(delta))) < 1e-9, "an uncentred bias makes the bound a lie"
    assert float(np.max(np.abs(delta))) <= MAX_ABS_BIAS + 1e-9


def test_absent_dimensions_contribute_nothing():
    """A dimension nothing answered for must not push the vocabulary."""
    head = _trained(seed=5)
    state = _covered_state()
    with_value = state.do(**{"goal.priority": 0.9})
    cleared = with_value.ablate("goal")
    assert not np.allclose(head.delta_logits(state), head.delta_logits(with_value))
    assert np.allclose(head.delta_logits(state), head.delta_logits(cleared))


def test_a_head_of_the_wrong_shape_is_rejected_at_construction():
    with pytest.raises(HeadUnusableError):
        EndogenousVocabHead(
            weights=np.zeros((10, STATE_DIM + 1), dtype=np.float32),
            bias=np.zeros(10, dtype=np.float32),
            vocab_size=10,
            layout=layout_digest(),
        semantics=semantics_digest(),
            tokenizer="sig",
            trained=True,
        )


def test_non_finite_weights_are_rejected():
    weights = np.zeros((4, STATE_DIM), dtype=np.float32)
    weights[0, 0] = np.inf
    with pytest.raises(HeadUnusableError):
        EndogenousVocabHead(
            weights=weights,
            bias=np.zeros(4, dtype=np.float32),
            vocab_size=4,
            layout=layout_digest(),
        semantics=semantics_digest(),
            tokenizer="sig",
            trained=True,
        )


def test_rebinding_to_another_model_keeps_the_weights_and_drops_the_claim():
    head = _trained()
    rebound = head.rebind(tokenizer="another-model")
    assert rebound.trained is False
    assert rebound.tokenizer == "another-model"
    assert np.allclose(rebound.weights, head.weights)
    _, decision = rebound.decide(_covered_state(), tokenizer_sig="another-model", alpha=1.0)
    assert decision.reason == "head_untrained"


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    head = _trained(vocab=32, seed=7)
    head.save(tmp_path)
    loaded = EndogenousVocabHead.load(tmp_path)
    assert loaded.tokenizer == head.tokenizer
    assert loaded.layout == head.layout
    assert loaded.trained is True
    assert np.allclose(loaded.weights, head.weights)


def test_loading_nothing_raises_a_reason_not_a_head(tmp_path):
    with pytest.raises(HeadUnusableError):
        EndogenousVocabHead.load(tmp_path / "nowhere")


def test_a_tokenizer_with_no_vocabulary_cannot_be_fingerprinted():
    with pytest.raises(HeadUnusableError):
        tokenizer_signature(object())


def test_two_tokenizers_of_the_same_size_get_different_fingerprints():
    """Size alone would let a head load against a model it was never fitted to."""

    class Tok:
        def __init__(self, offset: int) -> None:
            self._vocab = {f"tok{i + offset}": i for i in range(50)}

        def get_vocab(self):
            return self._vocab

    assert tokenizer_signature(Tok(0)) != tokenizer_signature(Tok(1))
    assert tokenizer_signature(Tok(0)) == tokenizer_signature(Tok(0))


def test_alpha_from_env_is_bounded(monkeypatch):
    monkeypatch.setenv("AURA_ENDOGENOUS_ALPHA", "not-a-number")
    assert alpha_from_env(0.6) == 0.6
    monkeypatch.setenv("AURA_ENDOGENOUS_ALPHA", "-3")
    assert alpha_from_env() == 0.0
    monkeypatch.setenv("AURA_ENDOGENOUS_ALPHA", "99")
    assert alpha_from_env() == 4.0
    monkeypatch.setenv("AURA_ENDOGENOUS_ALPHA", "nan")
    assert alpha_from_env() == 0.0
