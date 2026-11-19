"""What was said is read once, and each candidate next turn is scored after it (her turns' first reading).

LIVE 19 November 2026 a benchmark request of 47,125 characters waited 38 s before its prompt was built: the reading of
what it asks of her scored up to 64 candidate replies in one batch, each beside its own copy of the request, some 12,000
tokens a row. Now the request is read once into the model's cache, each candidate is scored after it and trimmed away,
and that is used only for a model on which it has been seen to give the batch's scores.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from core.brain.llm import what_she_expects_next as expects  # noqa: E402

CANDIDATES = ["Sure.", "No, I can't.", "What do you mean?", "Let me look."]


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 97 + 1 for char in str(text)]


@pytest.fixture
def tiny_model():
    """A Qwen2 of two layers and random weights, in float32: her readers' architecture, small enough to run anywhere."""
    from mlx_lm.models import qwen2

    mx.random.seed(7)
    model = qwen2.Model(qwen2.ModelArgs(model_type="qwen2", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
                                        num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=100, num_key_value_heads=2,
                                        max_position_embeddings=4096, rope_theta=10000.0, rope_traditional=False,
                                        tie_word_embeddings=True))
    mx.eval(model.parameters())
    expects._ONCE_AGREES.pop(id(model), None)
    yield model
    expects._ONCE_AGREES.pop(id(model), None)


def _batch(model, said):
    prefix = _Tokenizer().encode(expects.transcript_of(said))
    return expects._scored_in_one_batch(model, prefix, [_Tokenizer().encode(c) for c in CANDIDATES])


def test_reading_once_gives_the_batchs_scores(tiny_model):
    said = "Could you look up when the next train leaves?"
    once = expects._scored_after_one_reading(tiny_model, _Tokenizer().encode(expects.transcript_of(said)),
                                             [_Tokenizer().encode(c) for c in CANDIDATES])
    assert once == pytest.approx(_batch(tiny_model, said), rel=1e-4, abs=1e-4)


def test_a_request_longer_than_one_prefill_step_reads_the_same(tiny_model, monkeypatch):
    monkeypatch.setattr(expects, "_prefill_step", lambda: 16)
    said = "Plan the moves: " + " then ".join(f"put block {n} on block {n + 1}" for n in range(12))
    scores = expects.score_next_turns(tiny_model, _Tokenizer(), said, CANDIDATES)
    assert expects._ONCE_AGREES[id(tiny_model)] is True, "measured, and found to agree"
    assert scores == pytest.approx(_batch(tiny_model, said), rel=1e-4, abs=1e-4)


def test_a_model_that_keeps_no_cache_is_scored_in_one_batch_as_before():
    def flat(batch):
        rows, length = batch.shape
        return mx.broadcast_to(mx.arange(100).astype(mx.float32), (rows, length, 100))

    scores = expects.score_next_turns(flat, _Tokenizer(), "watch this with me?", CANDIDATES)
    assert len(scores) == len(CANDIDATES) and expects._ONCE_AGREES[id(flat)] is False


def test_a_model_on_which_reading_once_disagrees_keeps_the_batch(tiny_model, monkeypatch):
    monkeypatch.setattr(expects, "_scored_after_one_reading", lambda model, prefix, rows: [0.0] * len(rows))
    said = "Could you look up when the next train leaves?"
    assert expects.score_next_turns(tiny_model, _Tokenizer(), said, CANDIDATES) == \
        pytest.approx(_batch(tiny_model, said), rel=1e-6)
    assert expects._ONCE_AGREES[id(tiny_model)] is False
