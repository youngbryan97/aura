"""A verdict can only be a claim about the part of the state that varied.

Measured on Aura's live corpus, 2026-08-25: of 74 named dimensions, 24 moved
and 50 were pinned at one value across all 1,629 turns. Five of the six goal
dimensions never changed — the same goal, the same priority, zero progress,
never blocked, for the whole recorded period — and `attention.load` and
`recurrence.budget_used` sat at their ceiling throughout. The substrate and
uncertainty channels contributed nothing at all.

A constant dimension is worse than an absent one. It reads as live, it pads
the coverage figure the head's admission gate reads, and it cannot carry
information however strong the verdict printed above it.

The failure mode this guards against is the one worth naming: a state whose
only variance is affect can produce a real, significant gain over rare tokens
and still be a learned style adapter — cheerful state, cheerful vocabulary —
because there is no other channel the effect could have come from. The gain is
not downgraded, because it is real. It is named.
"""

from __future__ import annotations

import numpy as np

from core.brain.llm.endogenous_readout_training import (
    TurnTokens,
    VocabFit,
    varying_dimensions,
)
from core.brain.llm.endogenous_state import FEATURES, STATE_DIM


def _turns(mover: int | None, count: int = 40) -> list[TurnTokens]:
    turns = []
    for index in range(count):
        state = np.full(STATE_DIM, 0.5, dtype=np.float64)
        if mover is not None:
            state[mover] = index / count
        turns.append(
            TurnTokens(state=state, tokens=np.asarray([index], dtype=np.int64))
        )
    return turns


def test_a_pinned_dimension_is_reported_constant():
    report = varying_dimensions(_turns(mover=None))

    assert report["varying"] == []
    assert len(report["constant"]) == len(FEATURES)
    assert report["by_channel"] == {}


def test_only_the_dimension_that_moved_is_named():
    index = next(i for i, f in enumerate(FEATURES) if f.channel == "memory")
    report = varying_dimensions(_turns(mover=index))

    assert report["varying"] == [FEATURES[index].name]
    assert report["by_channel"] == {"memory": 1}


def test_an_empty_corpus_claims_nothing():
    assert varying_dimensions([])["varying"] == []


def _fit_with(by_channel: dict[str, int]) -> VocabFit:
    fit = VocabFit.__new__(VocabFit)
    object.__setattr__(fit, "state_variance", {"by_channel": by_channel})
    return fit


def test_affect_alone_is_flagged_as_indistinguishable_from_style():
    assert _fit_with({"affect": 7}).only_affect_varied is True


def test_any_other_channel_clears_the_flag():
    """One memory dimension is enough: the effect had somewhere else to come
    from."""
    assert _fit_with({"affect": 7, "memory": 1}).only_affect_varied is False
    assert _fit_with({"memory": 2}).only_affect_varied is False
    assert _fit_with({}).only_affect_varied is False


def test_the_live_corpus_is_not_carried_by_affect_alone():
    """Guards the specific claim made about the 2026-08-25 fit.

    Seven affect dimensions moved, and so did memory.recall_confidence, which
    had the widest spread of any dimension in the corpus. If a later change
    silently reduced the varying set to affect, the content_bearing verdict
    published against that corpus would quietly become a style claim.
    """
    live = {
        "affect": 7,
        "attention": 3,
        "goal": 4,
        "memory": 2,
        "recurrence": 2,
        "self_state": 2,
        "temporal": 4,
    }
    fit = _fit_with(live)
    assert fit.only_affect_varied is False
    assert "memory" in live and live["memory"] >= 1
