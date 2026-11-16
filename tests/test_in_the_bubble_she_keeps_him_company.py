"""In the bubble she is keeping the person company, and what is on screen reaches her as it does in any with.

Bryan, 15 November 2026: "unprompted advice and commentary. that's agency. just giving thoughts on the screen she
should be seeing when she feels its appropriate or has something to add ... the idea ultimately is for aura to always
have presence". Her remark at the desk was gated on a list of fault words, so the job application he was filling in
never reached her; in a with, whether to say something is her interest and the with's expected utility
(core/together/remarks.py, core/together/when_to_say_something.py).
"""
from __future__ import annotations

import asyncio
import time

import pytest

pytestmark = pytest.mark.unit

PERSON = "person:bryan"


@pytest.fixture
def presence(monkeypatch):
    from core.perception import ambient_presence
    from core.together import a_with

    monkeypatch.setattr(a_with, "_TOGETHER", a_with.Together())
    monkeypatch.setattr("core.together.taking_it_up.the_person_she_is_with", lambda: PERSON)
    monkeypatch.setattr(ambient_presence, "_proactivity_suppressed", lambda: False)
    routed: dict[str, object] = {}

    async def in_a_with(held, _observation, **kwargs):
        routed["with"] = held
        routed["kwargs"] = kwargs
        return "That second question wants the five layers of one thing you built, not five things."

    async def fault_gate(_result):
        routed["fault_gate"] = True
        return ""

    monkeypatch.setattr("core.together.remarks.remark_in_a_with", in_a_with)
    monkeypatch.setattr("core.perception.ambient_utterance.consider_utterance", fault_gate)
    monkeypatch.setattr("core.perception.ambient_utterance.latest_screen", lambda: object())

    async def kept(words, *, origin):
        routed.setdefault("kept", []).append((words, origin))
        return True

    monkeypatch.setattr("core.conversation.said_unasked.she_said_unasked", kept)
    held = ambient_presence.AmbientPresence()
    return held, routed


def test_in_the_bubble_what_she_sees_goes_to_the_company_she_is_keeping(presence):
    from core.perception.ambient_presence import KEEPING_COMPANY, PresenceMode

    held, routed = presence
    held.set_mode(PresenceMode.BUBBLE)

    asyncio.run(held._consider_speaking(None))

    assert routed["with"].who == PERSON and routed["with"].what == KEEPING_COMPANY
    from core.perception.ambient_utterance import compose_at_the_desk

    assert routed["kwargs"]["against_usual"] is True and routed["kwargs"]["compose"] is compose_at_the_desk
    assert "fault_gate" not in routed
    assert held.state()["has_utterance"] is True


def test_what_she_says_in_the_bubble_is_kept_in_her_conversation(presence):
    """LIVE 15 November 2026: a remark in the bubble was shown and kept nowhere, so her next turn could not see it."""
    from core.perception.ambient_presence import PresenceMode

    held, routed = presence
    held.set_mode(PresenceMode.BUBBLE)

    asyncio.run(held._consider_speaking(None))

    assert routed["kept"] == [(
        "That second question wants the five layers of one thing you built, not five things.", "ambient_remark")]


def test_her_window_opening_ends_the_company_kept_at_the_desk(presence):
    from core.perception.ambient_presence import PresenceMode
    from core.together.a_with import get_together

    held, _routed = presence
    held.set_mode(PresenceMode.BUBBLE)
    asyncio.run(held._consider_speaking(None))
    assert get_together().current(PERSON) is not None

    held.set_mode(PresenceMode.WINDOW)

    assert get_together().current(PERSON) is None


def test_with_her_window_up_only_the_fault_gate_reaches_her(presence):
    from core.perception.ambient_presence import PresenceMode

    held, routed = presence
    held.set_mode(PresenceMode.WINDOW)

    asyncio.run(held._consider_speaking(None))

    assert routed.get("fault_gate") is True and "with" not in routed


def test_the_person_at_the_desk_is_someone_she_knows_by_name():
    """LIVE 15 November 2026: the company she kept was held with "person:someone at desktop_email_reply"."""
    from types import SimpleNamespace

    from core.together.taking_it_up import who_she_is_with

    def person(key, last):
        return SimpleNamespace(kind="person", key=key, last=last, under_way={})

    record = SimpleNamespace(_things={
        "a": person("person:bryan", 2580),
        "b": person("person:someone at desktop_email_reply", 2593),
    })
    assert who_she_is_with(record, by_name=True) == "person:bryan"
    assert who_she_is_with(record) == "person:someone at desktop_email_reply"
    nobody_named = SimpleNamespace(_things={"b": person("person:someone at desktop_ui", 10)})
    assert who_she_is_with(nobody_named, by_name=True) == "person:someone at desktop_ui"


def test_a_with_begun_without_a_word_carries_on_from_the_sitting_going_on_now(monkeypatch):
    """LIVE 15 November 2026: she judged a remark worth making twice, and nothing was composed."""
    from core.together import remarks
    from core.together.a_with import With

    monkeypatch.setattr(remarks, "said_since", lambda began: [] if began > 0 else [
        (time.time() - 60.0, "user", "Not bad, slept okay."), (time.time() - 30.0, "assistant", "Glad to hear it.")])
    seen: dict[str, object] = {}

    class Router:
        async def think(self, _prompt, *, messages, **_kwargs):
            seen["messages"] = messages
            return "That second question wants one thing you built, five levels down."

    monkeypatch.setattr("core.container.ServiceContainer.get", lambda name, default=None: Router() if name == "llm_router" else default)
    held = With(who=PERSON, what="keeping company at the desk", began=100.0)

    said = asyncio.run(remarks._her_remark(held, "Describe something you built from scratch.", "TextEdit"))

    assert said.startswith("That second question")
    assert [m["role"] for m in seen["messages"]][:2] == ["user", "assistant"]
    assert "WHAT IS ON THE SCREEN" in seen["messages"][-1]["content"]


def test_a_highlight_nobody_asked_for_or_invited_is_still_refused():
    from core.perception.screen_highlight import highlight

    result = asyncio.run(highlight("Submit application"))

    assert not result.shown and "asked for" in result.reason


def test_at_the_desk_an_ordinary_look_is_worth_nothing_against_the_ones_before_it(monkeypatch):
    """LIVE 15 November 2026: a still screen cost nothing, and a remark worth 0.058 was made."""
    from core.together import remarks
    from core.together.a_with import With

    monkeypatch.setattr(remarks, "_interest", lambda capture, change: (0.058, ""))
    monkeypatch.setattr("core.together.taking_it_up._how_much_changed", lambda held, capture: 0.0)
    monkeypatch.setattr("core.together.taking_it_up._what_is_in_front", lambda capture: "TextEdit")
    composed: list[str] = []

    async def compose(_observation):
        composed.append("asked")
        return "A remark."

    held = With(who=PERSON, what="keeping company at the desk", began=0.0)
    for seen in (0.058, 0.058, 0.058):
        held.interests.append(seen)
    from types import SimpleNamespace

    said = asyncio.run(remarks.remark_in_a_with(held, SimpleNamespace(capture="ordinary", source="TextEdit"),
                                               now=100.0, compose=compose, against_usual=True))
    assert said == "" and composed == []


def test_her_own_recent_turn_said_again_is_not_a_remark(monkeypatch):
    from core.together import remarks

    monkeypatch.setattr(remarks, "said_since", lambda began: [
        (1.0, "user", "I slept badly last night, maybe four hours."),
        (2.0, "assistant", "I\u2019m sorry you\u2019re waking up on so little rest. That can make everything feel heavier today."),
    ])
    assert remarks._said_already("I\u2019m sorry you\u2019re waking up on so little rest.  That can make everything feel heavier today.")
    assert not remarks._said_already("That second question wants one thing you built, five levels down.")


def test_after_her_answer_what_she_notices_is_the_turn_before_hers_and_not_a_tool_she_called(monkeypatch):
    """LIVE 16 November 2026, 12:46 local: after "Can you highlight the back button?" and her answer, the reading of her
    host went in as runtime evidence, which her template writes as a tool's response, and she answered the question
    again: "I don't have the ability to visually highlight or draw on the screenshot." She had drawn the ring."""
    from core.brain.her_unprompted_turn import her_conversation_now_with
    from core.together import remarks
    from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE

    now = time.time()
    answered = [(now - 960.0, "user", "Can you highlight the back button?"),
                (now - 959.0, "assistant", "There: the arrow at the top left of Chrome.")]
    monkeypatch.setattr(remarks, "said_since", lambda began: list(answered))

    messages = her_conversation_now_with("THE HOST, READ NOW", "processor in use: 12%")

    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert "THE HOST, READ NOW" in messages[-1]["content"]

    monkeypatch.setattr(remarks, "said_since", lambda began: list(answered[:1]))
    waiting = her_conversation_now_with("THE HOST, READ NOW", "processor in use: 12%")
    assert waiting[-1]["role"] == RUNTIME_EVIDENCE_ROLE, "words still waiting for her keep the reading beside them"
