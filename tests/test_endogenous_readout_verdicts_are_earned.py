"""The trainer is run against corpora whose answer is known.

A detector that always finds signal is worth nothing, and a verdict that can
never be reached is worth less than nothing, because it reports green forever.
So three corpora are built here:

* **null** — states and tokens with no relationship whatsoever.
* **style** — one state dimension shifts the two most frequent tokens, the
  register effect a substrate genuinely can carry.
* **content** — one state dimension decides which of several mid-frequency
  words appears, without changing how often the pair appears at all.

The expected verdicts are ``no_signal``, ``style_prior`` and
``content_bearing``. Every measurement fault found while building this was
found by running exactly this battery: a rare-token gain compared against an
overall null called the style corpus content-bearing, a per-parameter
optimiser reported no signal on every corpus, and requiring overall
significance before a rare claim reported the content corpus as no signal.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.brain.llm.endogenous_readout_training import (
    MIN_TURNS,
    TurnTokens,
    fit_vocab_head,
    tokenize_pairs,
)
from core.brain.llm.endogenous_state import STATE_DIM

VOCAB = 160
#: Zipf-shaped, so the frequent end of the vocabulary behaves like function
#: words and the tail behaves like content words without anyone writing a list.
_BASE = 1.0 / np.arange(1, VOCAB + 1) ** 1.1
_BASE /= _BASE.sum()

#: Mid-frequency pairs: outside the frequent end, common enough to be learnable.
_PAIRS = ((40, 41), (50, 51), (60, 61), (70, 71), (45, 46), (55, 56))

#: The head is declared over a wider vocabulary than the corpus uses, which
#: is the real situation: a tokenizer holds a hundred thousand ids and a
#: corpus exercises a few thousand of them.
DECLARED_VOCAB = VOCAB + 40

#: Deliberately the state dimension the style corpus does not use, so a fit
#: cannot pass both corpora by keying on the same column.
_CONTENT_DIMENSION = 5
_STYLE_DIMENSION = 0


def _corpus(mode: str, seed: int, turns: int = 1500, tokens: int = 60):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(turns):
        state = np.clip(rng.normal(0.0, 0.6, STATE_DIM), -1.0, 1.0)
        probabilities = _BASE.copy()
        if mode == "style":
            shift = float(np.tanh(state[_STYLE_DIMENSION]))
            probabilities[0] *= 1.0 + 0.9 * shift
            probabilities[1] *= 1.0 - 0.9 * shift
        elif mode == "content":
            weight = 1.0 / (1.0 + np.exp(-6.0 * float(state[_CONTENT_DIMENSION])))
            for left, right in _PAIRS:
                mass = probabilities[left] + probabilities[right]
                probabilities[left] = mass * weight
                probabilities[right] = mass * (1.0 - weight)
        probabilities /= probabilities.sum()
        out.append(
            TurnTokens(
                state=state.astype(np.float64),
                tokens=rng.choice(VOCAB, size=tokens, p=probabilities).astype(np.int64),
            )
        )
    return out


def _fit(mode: str, seed: int = 1):
    return fit_vocab_head(
        _corpus(mode, seed),
        vocab_size=DECLARED_VOCAB,
        tokenizer_signature="synthetic",
        permutations=80,
        null_refits=1,
        seed=900 + seed,
        decays=(1e-2, 1e-1, 1.0),
    )


@pytest.fixture(scope="module")
def fits():
    return {mode: _fit(mode) for mode in ("null", "style", "content")}


def test_a_corpus_with_no_relationship_reports_no_signal(fits):
    fit = fits["null"]
    assert fit is not None
    assert fit.verdict == "no_signal"
    assert fit.usable is False


def test_a_register_effect_reports_a_style_prior(fits):
    fit = fits["style"]
    assert fit.verdict == "style_prior"
    assert fit.usable is True
    assert fit.improvement_nats > 0.0
    assert fit.overall_test.significant


def test_a_rare_word_identity_effect_reports_content_bearing(fits):
    """The strong verdict has to be reachable, or it reports green forever."""
    fit = fits["content"]
    assert fit.verdict == "content_bearing"
    assert fit.rare_test.significant
    assert fit.rare_improvement_nats > 0.0


def test_a_negative_gain_is_never_significant(fits):
    """A p-value below threshold with a gain below zero is still no finding."""
    fit = fits["style"]
    assert fit.rare_improvement_nats < 0.0
    assert fit.rare_test.significant is False


def test_the_report_carries_its_nulls_and_says_what_the_verdict_means(fits):
    report = fits["style"].as_report()
    assert report["permutation_tests"]["overall"]["permutations"] == 80
    assert report["permutation_tests"]["rare"]["permutations"] == 80
    assert report["what_this_means"]
    assert report["held_out"]["random_projection_baseline"]
    assert report["weight_decay_selected"] in (1e-2, 1e-1, 1.0)


def test_a_head_is_only_marked_trained_when_the_verdict_earns_it(fits):
    assert fits["null"].to_head().trained is False
    assert fits["style"].to_head().trained is True
    assert fits["style"].to_head().report["verdict"] == "style_prior"


def test_the_head_is_silent_about_tokens_the_corpus_never_contained(fits):
    head = fits["style"].to_head()
    active = set(int(t) for t in fits["style"].active_tokens)
    silent = [i for i in range(DECLARED_VOCAB) if i not in active]
    assert silent, "this corpus exercised the whole vocabulary; widen it"
    assert np.allclose(head.weights[silent], 0.0)


def test_too_few_turns_refuses_rather_than_fitting():
    assert (
        fit_vocab_head(
            _corpus("style", 1, turns=MIN_TURNS - 1),
            vocab_size=DECLARED_VOCAB,
            tokenizer_signature="synthetic",
            permutations=4,
            null_refits=0,
        )
        is None
    )


def test_tokenizing_needs_a_tokenizer():
    with pytest.raises(ValueError):
        tokenize_pairs([], object())
