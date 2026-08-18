"""Aura's own consent gate declining was reported as a broken pipeline.

99 "Fix generation or sandbox testing failed: Vetoed by entity" in one
sampled window, beside 57 "🚫 [GrowthLadder] Aura VETOED code repair".

The Growth Ladder is her consent gate on modifying her own source. When it
declines, repair_bug returns {"error": "Vetoed by entity"} and the engine
reported that as a failure — describing the mechanism working exactly as
designed. Nothing generated a bad fix and no sandbox failed. She said no.

Reported as failure it is worse than noise: a healthy refusal looks identical
to a broken repair pipeline, so the honest signal that the pipeline IS broken
has nowhere left to stand out.
"""
from __future__ import annotations

import inspect

import pytest

from core.self_modification import self_modification_engine as engine


def _failure_branch() -> str:
    source = inspect.getsource(engine)
    start = source.index("Fix generation or sandbox testing failed")
    return source[max(0, start - 1400) : start + 200]


def test_a_veto_is_not_logged_as_a_failure():
    branch = _failure_branch()

    assert '"veto" in reason.lower()' in branch, (
        "a veto must be told apart from a fix that could not be generated"
    )


def test_a_veto_is_reported_at_a_level_that_says_decision():
    branch = _failure_branch()
    veto_at = branch.index('"veto" in reason.lower()')
    handled = branch[veto_at : veto_at + 320]

    assert "logger.info" in handled
    assert "declined" in handled.lower()


def test_a_real_failure_is_still_a_warning():
    """Silencing the refusal is only worth doing if the fault stays loud."""
    branch = _failure_branch()

    assert 'logger.warning("Fix generation or sandbox testing failed' in branch


@pytest.mark.parametrize(
    ("reason", "is_veto"),
    [
        ("Vetoed by entity", True),
        ("VETOED by growth ladder", True),
        ("sandbox timeout", False),
        ("Unknown error", False),
    ],
)
def test_the_discriminator_matches_the_string_repair_actually_returns(reason, is_veto):
    """code_repair returns exactly "Vetoed by entity"; match must survive case."""
    assert ("veto" in reason.lower()) is is_veto


def test_the_veto_string_still_comes_from_code_repair():
    """If that literal changes, the classification silently stops working."""
    from core.self_modification import code_repair

    assert "Vetoed by entity" in inspect.getsource(code_repair)
