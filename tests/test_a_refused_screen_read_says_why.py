"""A screen reading that was refused says why, as a reading her model takes.

LIVE 15 November 2026, 22:22, a private Chrome window on screen: her screen reading said only "Screen capture was
refused for this turn.", and her draft said "I read the screen myself moments ago, and what I saw was a window
labeled 'Reading.'"
"""
from __future__ import annotations

import asyncio


def test_a_refused_screen_read_names_its_cause(monkeypatch):
    from core.brain import observable_registry
    from core.perception import screen_perception

    class _Refused:
        capture_denied = True
        unavailable_reason = "screen capture refused because private content is visible"

    class _Perception:
        async def capture(self, save_screenshot=False):
            return _Refused()

    monkeypatch.setattr(screen_perception, "get_screen_perception", lambda: _Perception())
    reading = asyncio.run(observable_registry._read_screen("What is on my screen right now?"))
    assert reading == ("Nothing was read from the screen for this turn: "
                       "screen capture refused because private content is visible.")


def test_only_the_corpus_and_page_work_stand_for_the_world():
    from core.brain.observable_grounding import OBSERVABLES

    assert {o.name for o in OBSERVABLES if o.of_the_world} == {"corpus", "page_work_after_this_reply"}
