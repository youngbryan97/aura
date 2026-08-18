"""Capability present, content never supplied — the defect this registry closes.

Four incidents, one shape. Each was fixed on its own before the pattern was
obvious:

    "how many .py files in core/introspection"  ->  "There are 3"      (ten)
    "read CONTRIBUTING.md"                      ->  "I tried and failed"
                                                    (nothing ran)
    "what's on my clipboard right now?"         ->  "I can only work with the
                                                     information you provide"
                                                    (it held BUILD-7741-verify)
    "what time is it"                           ->  "my clock says 06:15 and the
                                                     ambient light sensors report
                                                     low illumination"  (at 01:40,
                                                     with no light sensor)

The capability was registered every time and the reading was cheap every time.
Nothing took the reading before the answer was composed, and a model asked
about a fact it does not hold produces something fact-shaped.
"""

from __future__ import annotations

import asyncio

import pytest

import core.brain.observable_registry as registry
from core.brain.observable_grounding import (
    Observable,
    OBSERVABLES,
    observable_blocks,
    register_observable,
)


def _blocks(prompt: str) -> list[str]:
    return asyncio.run(observable_blocks(prompt))


# ── the registry is populated and routes ────────────────────────────────────

def test_the_expected_observables_are_registered() -> None:
    names = set(registry.observable_names())

    assert {"clipboard", "file", "file_count", "corpus", "clock"} <= names


@pytest.mark.parametrize(
    ("prompt", "header"),
    [
        ("read CONTRIBUTING.md", "## FILE YOU WERE ASKED ABOUT"),
        (
            "how many python files are in core/introspection",
            "## DIRECTORY LISTING YOU WERE ASKED ABOUT",
        ),
        ("what time is it?", "## THE CURRENT LOCAL TIME"),
    ],
)
def test_a_question_gets_its_reading(prompt: str, header: str) -> None:
    blocks = _blocks(prompt)

    assert any(block.startswith(header) for block in blocks), blocks


def test_a_conversational_turn_takes_no_readings() -> None:
    """Readers are privacy- and latency-relevant; they are never ambient."""
    assert _blocks("how are you doing today?") == []


def test_the_directory_listing_carries_the_real_count() -> None:
    from pathlib import Path

    truth = len(list((Path(__file__).resolve().parents[2] / "core" / "introspection").glob("*.py")))
    blocks = _blocks("how many python files are in core/introspection")

    assert any(str(truth) in block for block in blocks)


# ── the mechanism's own contract ────────────────────────────────────────────

def test_a_failing_reader_does_not_break_the_turn() -> None:
    async def _boom(_prompt: str) -> str:
        raise OSError("device unavailable")

    register_observable(
        Observable("test_broken", "## BROKEN", lambda _p: True, _boom)
    )
    try:
        assert _blocks("anything at all") == [] or all(
            "## BROKEN" not in block for block in _blocks("anything")
        )
    finally:
        OBSERVABLES[:] = [o for o in OBSERVABLES if o.name != "test_broken"]


def test_a_slow_reader_cannot_hold_the_turn() -> None:
    async def _slow(_prompt: str) -> str:
        await asyncio.sleep(5)
        return "too late"

    register_observable(
        Observable("test_slow", "## SLOW", lambda _p: True, _slow, timeout_s=0.15)
    )
    try:
        import time

        started = time.monotonic()
        blocks = _blocks("anything")
        elapsed = time.monotonic() - started

        assert elapsed < 3.0
        assert all("## SLOW" not in block for block in blocks)
    finally:
        OBSERVABLES[:] = [o for o in OBSERVABLES if o.name != "test_slow"]


def test_a_raising_matcher_is_survivable() -> None:
    def _bad_matcher(_prompt: str) -> bool:
        raise ValueError("matcher exploded")

    async def _never(_prompt: str) -> str:
        return "unreachable"

    register_observable(Observable("test_bad", "## BAD", _bad_matcher, _never))
    try:
        assert all("## BAD" not in block for block in _blocks("anything"))
    finally:
        OBSERVABLES[:] = [o for o in OBSERVABLES if o.name != "test_bad"]


def test_an_empty_reading_produces_no_block() -> None:
    async def _empty(_prompt: str) -> str:
        return "   "

    register_observable(Observable("test_empty", "## EMPTY", lambda _p: True, _empty))
    try:
        assert all("## EMPTY" not in block for block in _blocks("anything"))
    finally:
        OBSERVABLES[:] = [o for o in OBSERVABLES if o.name != "test_empty"]


def test_two_observables_in_one_question_both_answer() -> None:
    blocks = _blocks(
        "what's on my clipboard, and how many python files are in core/introspection?"
    )
    headers = {block.split("\n")[0] for block in blocks}

    assert "## DIRECTORY LISTING YOU WERE ASKED ABOUT" in headers
    assert "## WHAT IS ON THE CLIPBOARD" in headers


def test_registering_the_same_name_replaces_rather_than_duplicates() -> None:
    async def _one(_prompt: str) -> str:
        return "first"

    async def _two(_prompt: str) -> str:
        return "second"

    register_observable(Observable("test_dup", "## DUP", lambda _p: True, _one))
    register_observable(Observable("test_dup", "## DUP", lambda _p: True, _two))
    try:
        dup = [block for block in _blocks("anything") if block.startswith("## DUP")]

        assert len(dup) == 1
        assert "second" in dup[0]
    finally:
        OBSERVABLES[:] = [o for o in OBSERVABLES if o.name != "test_dup"]


# ── screen and beliefs, the next two of the forty-three ─────────────────────
#
# "what's on my screen right now?" was answered "I couldn't get to an answer
# I'd stand behind on that one" while screen capture was permitted and working.
# "what do you currently believe about me?" was answered from the model while a
# belief graph sat unread.

def test_screen_questions_reach_the_screen_reader() -> None:
    blocks = _blocks("what's on my screen right now?")

    assert any(block.startswith("## WHAT IS ON THE SCREEN") for block in blocks)


@pytest.mark.parametrize(
    "prompt",
    [
        "what's on my screen right now?",
        "what do you see?",
        "which app is in front?",
        "what window am I looking at",
    ],
)
def test_screen_phrasings_are_recognised(prompt: str) -> None:
    from core.brain.observable_registry import _matches_screen

    assert _matches_screen(prompt) is True


def test_the_screen_reading_is_always_definite() -> None:
    """Named app, named absence, or named refusal — never silence.

    Screen capture admission legitimately refuses in some contexts (it does
    under pytest), and a refusal IS the answer to "what is on my screen". What
    must never happen is the empty block that let "I couldn't get to an answer
    I'd stand behind" stand in for a reading nobody took.
    """
    import asyncio

    from core.brain.observable_registry import _read_screen

    body = asyncio.run(_read_screen("what is on my screen"))

    assert body.strip()
    assert (
        "Frontmost application:" in body
        or "refused" in body.lower()
    ), body


def test_an_unreadable_window_names_the_absence(monkeypatch) -> None:
    """This is what stops 'the room is silent, the light unchanged' appearing."""
    import asyncio

    class _Snapshot:
        capture_denied = False
        active_app = "Safari"
        text = ""
        accessibility_text = ""
        focused_role = ""
        focused_name = ""

    class _Perception:
        async def capture(self, save_screenshot=False):
            return _Snapshot()

    monkeypatch.setattr(
        "core.perception.screen_perception.get_screen_perception", lambda: _Perception()
    )

    from core.brain.observable_registry import _read_screen

    body = asyncio.run(_read_screen("what is on my screen"))

    assert "Safari" in body
    assert "No readable text" in body


def test_a_refused_capture_is_reported_as_refused(monkeypatch) -> None:
    import asyncio

    class _Snapshot:
        capture_denied = True

    class _Perception:
        async def capture(self, save_screenshot=False):
            return _Snapshot()

    monkeypatch.setattr(
        "core.perception.screen_perception.get_screen_perception", lambda: _Perception()
    )

    from core.brain.observable_registry import _read_screen

    assert "refused" in asyncio.run(_read_screen("what is on my screen")).lower()


@pytest.mark.parametrize(
    "prompt",
    [
        "what do you currently believe about me?",
        "what do you think about me?",
        "tell me your beliefs",
    ],
)
def test_belief_questions_are_recognised(prompt: str) -> None:
    from core.brain.observable_registry import _matches_beliefs

    assert _matches_beliefs(prompt) is True


def test_a_conversational_turn_does_not_read_the_screen() -> None:
    """Screen capture is privacy-relevant; it is never ambient."""
    from core.brain.observable_registry import _matches_screen

    assert _matches_screen("how are you doing today?") is False


def test_every_registered_observable_is_visible_at_dispatch() -> None:
    """A hand-written copy of this list drifted the first day it existed.

    screen and beliefs were registered and left out of the survival check, so a
    screen reading that WAS taken reported as not surviving — and an hour went
    into hunting a delivery bug that did not exist.
    """
    from core.brain.inference_gate import _observable_dispatch_markers

    markers = dict(_observable_dispatch_markers())

    for observable in OBSERVABLES:
        assert observable.name in markers, observable.name
        assert markers[observable.name] == observable.header
