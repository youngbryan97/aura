"""The whole chain, with a head on disk.

Every other test in this pathway exercises one link. This one runs the chain:
a corpus of turns is fitted, the head is written and read back, a job carries
z_Aura across the process boundary the way the worker receives it, and the
bias that comes out is applied to logits and moved by an intervention.

The head here is fitted to a constructed corpus and is never written where the
runtime would find it. What it proves is that the links join, not that Aura's
substrate steers her words.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.brain.llm.cognitive_code import read_code
from core.brain.llm.endogenous_decode import (
    JOB_STATE_KEY,
    build_endogenous_processor,
    load_head,
    reset_head_cache,
)
from core.brain.llm.endogenous_intervention import (
    channel_influence_map,
    measure_contrast,
)
from core.brain.llm.endogenous_readout_training import TurnTokens, fit_vocab_head
from core.brain.llm.endogenous_state import STATE_DIM, assemble_state
from core.brain.llm.endogenous_vocab_head import EndogenousVocabHead

VOCAB = 120
DECLARED_VOCAB = VOCAB + 20
_BASE = 1.0 / np.arange(1, VOCAB + 1) ** 1.1
_BASE /= _BASE.sum()

#: The dimension the corpus is built around, so the fitted head has a known
#: channel to be sensitive to.
DRIVER = "uncertainty.confidence"


class _Tokenizer:
    """Just enough tokenizer to be fingerprinted and to encode."""

    def get_vocab(self):
        return {f"tok{i}": i for i in range(DECLARED_VOCAB)}

    def encode(self, text: str):
        return [int(piece) for piece in str(text).split() if piece.isdigit()]


def _corpus(turns: int = 900, tokens: int = 50, seed: int = 3):
    """Turns whose register depends on one named dimension of z_Aura."""
    from core.brain.llm.endogenous_state import FEATURE_INDEX

    driver = FEATURE_INDEX[DRIVER]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(turns):
        state = np.clip(rng.normal(0.0, 0.5, STATE_DIM), -1.0, 1.0)
        state[driver] = float(rng.uniform(0.0, 1.0))
        probabilities = _BASE.copy()
        shift = 2.0 * state[driver] - 1.0
        probabilities[0] *= 1.0 + 0.95 * shift
        probabilities[1] *= 1.0 - 0.95 * shift
        probabilities /= probabilities.sum()
        out.append(
            TurnTokens(
                state=state.astype(np.float64),
                tokens=rng.choice(VOCAB, size=tokens, p=probabilities).astype(np.int64),
            )
        )
    return out


@pytest.fixture(scope="module")
def fitted():
    return fit_vocab_head(
        _corpus(),
        vocab_size=DECLARED_VOCAB,
        tokenizer_signature="",
        permutations=60,
        null_refits=1,
        seed=41,
        decays=(1e-2, 1e-1, 1.0),
    )


@pytest.fixture
def head_on_disk(fitted, tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    from core.brain.llm.endogenous_vocab_head import tokenizer_signature

    tokenizer = _Tokenizer()
    head = fitted.to_head()
    bound = EndogenousVocabHead(
        weights=head.weights,
        bias=head.bias,
        vocab_size=head.vocab_size,
        layout=head.layout,
        tokenizer=tokenizer_signature(tokenizer),
        trained=head.trained,
        report=head.report,
        trained_at=head.trained_at,
    )
    bound.save(tmp_path / "head")
    reset_head_cache()
    yield tmp_path / "head", tokenizer
    reset_head_cache()


def test_the_corpus_fits_and_the_verdict_is_usable(fitted):
    assert fitted is not None
    assert fitted.verdict in {"style_prior", "content_bearing"}
    assert fitted.to_head().trained is True


def test_the_head_round_trips_through_disk(head_on_disk):
    directory, _tokenizer = head_on_disk
    loaded, reason = load_head(directory)
    assert loaded is not None, reason
    assert loaded.trained is True


def _live_state(confidence: float = 0.9):
    """A state with enough live channels to clear the coverage floor.

    The floor exists so a bias is never computed from mostly-absence, and a
    state answering four of seventy-four dimensions is mostly absence.
    """
    return assemble_state(
        probes={
            "uncertainty": lambda: {DRIVER: confidence},
            "goal": lambda: {"goal.active": 1.0, "goal.priority": 0.5},
            "memory": lambda: {"memory.recall_hits": 0.4},
            "substrate": lambda: {
                f"substrate.band_{i:02d}": 0.1 * ((i % 5) - 2) for i in range(32)
            },
        }
    )


def test_a_job_carrying_the_state_gets_a_processor(head_on_disk):
    directory, tokenizer = head_on_disk
    state = _live_state()
    processor, receipt = build_endogenous_processor(
        tokenizer, {JOB_STATE_KEY: state.to_payload()}, directory=directory
    )
    assert processor is not None, receipt
    assert receipt["applied"] is True
    assert receipt["nonzero_tokens"] > 0
    assert receipt["max_abs_delta"] > 0.0


def test_the_bias_changes_the_logits_it_is_allowed_to_change(head_on_disk):
    directory, tokenizer = head_on_disk
    state = _live_state()
    processor, receipt = build_endogenous_processor(
        tokenizer, {JOB_STATE_KEY: state.to_payload()}, directory=directory
    )
    assert processor is not None, receipt
    logits = np.zeros(processor.vocab_size)
    logits[5] = -50.0
    out = processor.apply_numpy(logits)
    assert not np.allclose(out, logits), "a trained head changed nothing"
    assert out[5] == logits[5], "a ruled-out token was touched"


def test_moving_the_driver_moves_the_bias_more_than_its_peers(head_on_disk):
    """The dimension the corpus was built around must lead its own nulls."""
    directory, _tokenizer = head_on_disk
    head, reason = load_head(directory)
    assert head is not None, reason
    effect = measure_contrast(_live_state(0.5), DRIVER, 0.05, 0.95, head=head)
    assert effect.bias_shift > 0.0
    assert effect.exceeds_null is True
    assert "UNCERTAINTY" in effect.code_lines_moved


def test_the_influence_map_finds_the_channel_the_corpus_used(head_on_disk):
    directory, _tokenizer = head_on_disk
    head, _reason = load_head(directory)
    influence = channel_influence_map(_live_state(), head)
    assert "uncertainty" in influence["channels_with_influence"]
    top = influence["channels"][0]
    assert top["channel"] == "uncertainty"


def test_the_code_still_refuses_to_be_shown(head_on_disk):
    assert read_code(_live_state(), include_organ_lines=False).is_user_presentable is False
