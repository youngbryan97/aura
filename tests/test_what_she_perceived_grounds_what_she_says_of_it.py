"""What she perceived on a turn is evidence she was given, as the conversation is.

LIVE 7 November 2026, dress rehearsal: her last word on a video she had watched with him was refused as a fabricated
shared past against no grounding at all, the look she had just taken of the video left out.
"""

import time

import pytest

from core.conversation import response_reliability
from core.senses.turn_evidence import TurnSensoryEvidence

ASKED = "Let's watch this one together and tell me what you make of the part about the lake birds."
SAID = "Remember when we watched the grey heron spear that silver trout off the jetty at dawn?"
SEEN = "A grey heron spears a silver trout off a jetty at dawn; the narrator says it waited an hour."


def _look(observation, *, ok=True):
    return TurnSensoryEvidence(channel="screen", ok=ok, request="", observed_at=time.time(), observation=observation,
                               source="watching_along", scope="what plays on her own page", coverage="partial").to_dict()


@pytest.fixture
def held_against(monkeypatch):
    seen: list[list[str]] = []

    def record(raw, user_message, recent, *, grounding=None):
        seen.append(list(grounding or ()))
        return False

    monkeypatch.setattr(response_reliability, "has_fabricated_shared_history", record)
    return seen


def test_what_she_saw_on_the_turn_is_part_of_what_a_shared_past_is_held_against(held_against):
    response_reliability.assess_user_facing_reply(ASKED, SAID, sensory_evidence=_look(SEEN))
    assert held_against and SEEN in held_against[-1]


def test_a_look_that_failed_is_not(held_against):
    response_reliability.assess_user_facing_reply(ASKED, SAID, sensory_evidence=_look(SEEN, ok=False))
    assert held_against and SEEN not in held_against[-1]


def test_no_look_adds_nothing(held_against):
    response_reliability.assess_user_facing_reply(ASKED, SAID)
    assert held_against and SEEN not in held_against[-1]
