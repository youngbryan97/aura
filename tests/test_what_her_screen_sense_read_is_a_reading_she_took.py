"""What her screen sense last read goes beside her turn as a reading, and her reply is held to it as one.

LIVE 15 November 2026, 22:43, a private window up and this turn's capture refused: her prompt carried "You looked at
the screen yourself moments ago. If asked where this came from, say you read the screen." with the windows her
ambient sense had read, her draft said "I can see you're using Claude, with a window open that shows 'Aura Zenith'",
and her reply check refused it as a screen she had not read. The note told her model what to say (F06), and nothing
let the check count the reading.
"""
from __future__ import annotations

import time

from core.perception.observation_evidence import (
    SENSORY_BRIEF_BANNER,
    Observation,
    ObservationKind,
    ObservationMemory,
)


def _memory_with(capture: str) -> ObservationMemory:
    memory = ObservationMemory()
    memory.record(Observation(kind=ObservationKind.SCREEN_TEXT, capture=capture, source="Claude",
                              at=time.time() - 30))
    return memory


def test_the_brief_is_a_reading_with_its_time_and_says_nothing_to_her_model():
    brief = _memory_with("Claude\nAura Zenith\nWhat is on my screen right now?").sensory_brief()
    assert brief.startswith(f"[{SENSORY_BRIEF_BANNER}]\nRead by her screen sense at ")
    assert " s before this turn." in brief
    for instruction in ("You looked", "If asked", "say you", "do not state", "NOTES, NOT A REPLY"):
        assert instruction not in brief, instruction


def test_the_brief_given_to_a_turn_licenses_her_saying_what_it_read():
    from core.conversation.response_reliability import _quotes_a_screen_it_did_not_read
    from core.conversation.turn_evidence_custody import (
        READ_ON_HER_SCREEN,
        bind_turn_evidence_custody,
        turn_world_sources,
    )

    memory = _memory_with('Claude — "Aura Zenith"\nThe build is green')
    with bind_turn_evidence_custody(session_id="s", turn_id="t-brief"):
        brief = memory.sensory_brief_for_this_turn()
        assert brief and turn_world_sources()[-1]["origin"].startswith(READ_ON_HER_SCREEN)
        assert not _quotes_a_screen_it_did_not_read(
            "What is on my screen right now?", 'The window in front says "The build is green".')
    with bind_turn_evidence_custody(session_id="s", turn_id="t-none"):
        assert _quotes_a_screen_it_did_not_read(
            "What is on my screen right now?", 'The window in front says "The build is green".'), (
            "with nothing read for the turn, a quotation of the screen is still refused")
