"""Token sentinel: a config value that crashed generation, and a 'clean' prefix
that kept the loop."""
from __future__ import annotations

import pytest

from core.brain.llm.token_sentinel import InterventionType, TokenSentinel

pytestmark = pytest.mark.unit


def _feed(sentinel: TokenSentinel, tokens):
    signal = None
    for tok in tokens:
        signal = sentinel.feed(tok)
        if signal is not None and signal.type is not InterventionType.NONE:
            return signal
    return signal


# ── config must not be able to kill generation ─────────────────────────────


@pytest.mark.parametrize("bad", [0, -5, None, "eight", 2.5])
def test_malformed_intervals_do_not_crash_generation(bad):
    """These are modulo divisors on EVERY token: a zero raised
    ZeroDivisionError straight out of the token loop, taking down generation
    from a config value."""
    sentinel = TokenSentinel(check_interval=bad, affect_interval=bad)

    for i in range(40):
        sentinel.feed(f"tok{i} ")

    assert sentinel._check_interval >= 1
    assert sentinel._affect_interval >= 1
    assert sentinel.get_diagnostics()["interval_corrected"] is True


def test_valid_intervals_are_kept():
    sentinel = TokenSentinel(check_interval=4, affect_interval=9)

    assert sentinel._check_interval == 4
    assert sentinel._affect_interval == 9
    assert sentinel.get_diagnostics()["interval_corrected"] is False


# ── the clean prefix must actually be clean ────────────────────────────────


def test_clean_prefix_removes_the_whole_loop():
    """seq_len and repeats count TOKENS; their product was used to slice
    CHARACTERS. Tokens are several characters wide, so the 'clean' prefix kept
    most of the loop it claimed to have removed."""
    sentinel = TokenSentinel(check_interval=1)
    good = "The answer is straightforward. "
    sentinel.feed(good)
    signal = _feed(sentinel, ["ABCDEFGH "] * 40)

    assert signal is not None
    assert signal.type is InterventionType.ABORT_LOOP
    # The whole repeated span is gone, not a character-count approximation.
    assert "ABCDEFGH" not in signal.clean_prefix
    assert signal.clean_prefix.strip() == good.strip()


def test_clean_prefix_is_empty_when_everything_looped():
    """A loop that spans the entire generation leaves nothing salvageable."""
    sentinel = TokenSentinel(check_interval=1)
    signal = _feed(sentinel, ["LOOPING "] * 40)

    assert signal is not None
    assert signal.type is InterventionType.ABORT_LOOP
    assert "LOOPING" not in signal.clean_prefix


# ── absences must be visible, not inferred ─────────────────────────────────


def test_inactive_live_affect_is_reported_not_silent():
    """The advertised live-affect path can be entirely inactive while
    diagnostics merely show zero pulses — indistinguishable from 'no pulse was
    due yet'."""
    sentinel = TokenSentinel(affect_interval=2, substrate_mem=None,
                             steering_hooks=None)

    for i in range(10):
        sentinel.feed(f"t{i} ")

    diagnostics = sentinel.get_diagnostics()
    assert diagnostics["live_affect_available"] is False
    assert diagnostics["affect_pulses"] == 0


def test_intentionally_neutral_generation_has_no_missing_steering_fault(monkeypatch):
    degradations = []
    monkeypatch.setattr(
        "core.brain.llm.token_sentinel.record_degradation",
        lambda *args, **kwargs: degradations.append((args, kwargs)),
    )
    sentinel = TokenSentinel(
        affect_interval=1,
        substrate_mem=None,
        steering_hooks=[],
        affect_expected=False,
    )

    sentinel.feed("one")
    sentinel.feed("two")

    assert degradations == []
    assert sentinel.get_diagnostics()["live_affect_expected"] is False


def test_ontology_degradation_is_visible(monkeypatch):
    """An import/runtime failure was folded into an empty match, so generation
    continued as though the semantic check had PASSED when it never ran."""
    import core.conversation.ontology_grounding as og

    def _boom(text):
        raise RuntimeError("grounder unavailable")

    monkeypatch.setattr(og, "detect_unsupported_embodiment_claim", _boom)

    sentinel = TokenSentinel(check_interval=1)
    for tok in "I walked to the store today and it was pleasant ".split():
        sentinel.feed(tok + " ")

    assert sentinel.get_diagnostics()["ontology_check_degraded"] is True
