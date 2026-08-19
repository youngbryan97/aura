"""A pursuit runs for minutes, so the owner has to be able to watch it.

Two things make that true and both are easy to lose. The window has to be on
screen, and what she chose has to reach the thought stream while she is
choosing it rather than in a trace handed over at the end. A headless run of a
long task is indistinguishable from a hung one.
"""

from __future__ import annotations

import inspect

from core.skills.sovereign_browser import SovereignBrowserSkill


def test_a_pursuit_opens_a_window_and_a_search_does_not():
    """Momentary reads stay out of the way; work the owner asked for is seen."""
    source = inspect.getsource(SovereignBrowserSkill)
    assert 'visible=(params.mode == "pursue")' in source, (
        "pursue must open a visible browser: a long task nobody can see is a "
        "long task nobody can stop"
    )
    signature = inspect.signature(SovereignBrowserSkill._create_browser)
    assert signature.parameters["visible"].default is False, (
        "search and browse must stay headless"
    )


def test_each_round_is_narrated_as_it_happens():
    narrate = getattr(SovereignBrowserSkill, "_narrate", None)
    assert narrate is not None, "a pursuit must be able to say what it is doing"
    body = inspect.getsource(narrate)
    assert "get_emitter" in body, "narration must reach the thought stream"
    for field in ("chose", "asked", "why"):
        assert f'"{field}"' in body, f"narration must carry {field}"

    loop = inspect.getsource(SovereignBrowserSkill._handle_pursue)
    assert "self._narrate(steps[-1])" in loop, (
        "narration must fire inside the loop, not after it"
    )


def test_narration_can_never_break_the_pursuit():
    """Saying what she is doing is never worth losing the work over."""
    body = inspect.getsource(SovereignBrowserSkill._narrate)
    assert "except Exception" in body and "record_degradation" in body, (
        "a failed emit must degrade, not raise, and must not be silent"
    )


def test_a_failed_round_does_not_destroy_the_finished_ones():
    """One slow generation used to take a whole run with it.

    A decision timed out at round forty-one, the exception left the loop, and
    forty landed answers went with it — reported as a failure that had done
    nothing.
    """
    body = inspect.getsource(SovereignBrowserSkill._handle_pursue)
    assert "pursuit_interrupted" in body, (
        "an unexpected failure must end the loop and report the work"
    )
    assert "landed_total" in body, "work that landed must be counted"
    assert 'action="final observation skipped"' in body, (
        "the last look must be best-effort: if the browser is what broke, the "
        "run still has everything it did before that"
    )


def test_a_pursuit_tells_the_watchdog_it_is_alive():
    body = inspect.getsource(SovereignBrowserSkill._handle_pursue)
    assert "report_progress" in body and "heartbeat" in body, (
        "without a heartbeat a pursuit is capped at the silence ceiling "
        "however much progress it is making"
    )
