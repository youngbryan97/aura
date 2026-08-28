"""Four deadlines wrap one turn, and only one of them owns it.

LIVE, 2026-08-28: a tool loop reached code_repl, the code raised NameError for
a missing import, the error was handed back for the model to fix — and the turn
was cut before it could. The inference gate had raised its own deadline to 345
seconds and the turn still ended at 186, because the CognitiveEngine cycle
timeout was a flat 180 set before anything knew what the answer would cost, and
everything else is nested inside it.

Extending one clock inside another is not a fix; it is the next whack. What
makes them agree is that each takes its floor from the same reading: what this
request's answer costs to decode, at the rate this machine has been measured
at. An unmeasured rate raises nothing, everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.brain.llm import thinking_reserve


@pytest.fixture(autouse=True)
def _measured(tmp_path, monkeypatch):
    """A known decode rate, so the arithmetic is checkable."""

    monkeypatch.setattr(thinking_reserve, "_store_path", lambda: tmp_path / "d.json")
    thinking_reserve.forget()
    for _ in range(20):
        # Six tokens a second, over runs long enough to speak about long ones.
        thinking_reserve.record_decode_rate(generated_tokens=900, elapsed_s=150.0)
    yield
    thinking_reserve.forget()


def test_every_clock_reads_the_same_measurement() -> None:
    """Three files, one reading. A fourth clock cannot quietly use its own."""

    for path, name in (
        ("core/brain/inference_gate.py", "_seconds_to_decode"),
        ("core/brain/llm_health_router.py", "_seconds_a_budget_needs"),
        ("core/brain/cognitive_engine.py", "_time_the_answer_needs"),
    ):
        body = Path(path).read_text()
        assert name in body, f"{path} has no reader"
        start = body.index(f"def {name}(")
        window = body[start : start + 1400]
        assert "seconds_to_decode" in window, f"{name} does not use the measurement"


def test_the_owning_clock_scales_with_what_the_answer_costs() -> None:
    from core.brain.cognitive_engine import _time_the_answer_needs

    plain = _time_the_answer_needs("how are you doing?")
    reasoning = _time_the_answer_needs(
        "Work out the rule. [1,2] becomes [2,1]. [3,4,5] becomes [5,4,3]. "
        "What does [6,7,8,9] become, why, and how sure are you?"
    )
    assert 0.0 < plain < reasoning


def test_a_turn_that_must_fetch_is_given_two_generations() -> None:
    """A call is a whole generation before the answer is started."""

    from core.brain.cognitive_engine import _time_the_answer_needs

    here = os.getcwd()
    fetching = _time_the_answer_needs(
        f"read the docs at {here}/README.md then actually use it and tell me what happened"
    )
    not_fetching = _time_the_answer_needs(
        "read the docs then actually use it and tell me what happened"
    )
    assert fetching > not_fetching


def test_an_unmeasured_rate_raises_nothing() -> None:
    from core.brain.cognitive_engine import _time_the_answer_needs
    from core.brain.llm_health_router import _seconds_a_budget_needs

    thinking_reserve.forget()
    assert _time_the_answer_needs("anything at all, at some length") == 0.0
    assert _seconds_a_budget_needs(1024) == 0.0


def test_the_endpoint_cap_cannot_be_shorter_than_the_budget_needs() -> None:
    """The argument was already written in this file, for long answers only.

    "Replacing it here with the ordinary 150-second cap makes the requested
    token budget impossible to consume and turns healthy slow decoding into a
    false endpoint failure." At the measured rate 1,024 tokens takes about 171
    seconds, and the branch above that comment capped the call at 150.
    """

    from core.brain.llm_health_router import _seconds_a_budget_needs

    needed = _seconds_a_budget_needs(1024)
    assert needed > 150.0

    body = Path("core/brain/llm_health_router.py").read_text()
    start = body.index("needed_s = _seconds_a_budget_needs(max_tokens)")
    window = body[start : start + 300]
    assert "cap_s = max(cap_s, needed_s + 2.0)" in window
