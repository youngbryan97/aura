"""A failure that is a standing condition is not a surprise the second time.

LIVE 2026-09-23: "Reddit inbox unavailable; login required (surprise=1.00)"
every forty-five minutes all day, each one feeding frustration. The caller's
expectation is written before anything is tried and is nearly always "it
works"; her own record of the last attempt is the better one.
"""

from __future__ import annotations

from core.agency.intention_loop import IntentionLoop

TRYING = "Read connected Reddit state for bounded autonomous social awareness."
EXPECTED = "Successful execution of reddit_adapter"
FAILED = "Reddit inbox unavailable; login required"


def _attempt(loop: IntentionLoop, outcome: str) -> float:
    intention_id = loop.intend(TRYING, drive="autonomous_initiative_loop", expected_outcome=EXPECTED)
    return loop.observe(intention_id, observation=outcome, actual_outcome=outcome)


def test_the_same_failure_again_is_expected(tmp_path):
    loop = IntentionLoop(db_path=str(tmp_path / "intentions.db"))
    first = _attempt(loop, FAILED)
    again = _attempt(loop, FAILED)
    assert first > 0.5
    assert again < first
    assert again <= 0.1


def test_a_different_failure_is_still_news(tmp_path):
    loop = IntentionLoop(db_path=str(tmp_path / "intentions.db"))
    _attempt(loop, FAILED)
    other = _attempt(loop, "Failed to load r/philosophy: connection reset by peer")
    assert other > 0.5
