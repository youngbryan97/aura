"""Watching a video with somebody is something the two of them are doing, beside the conversation: her answer to the
request goes first, then it opens; a remark waits while either of them has the floor or a turn of theirs is being
answered; every turn of the conversation knows what is playing; the with ends when the video does. Her eyes, her
turns, the player and the person are stand-ins here."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

ASKED = "Do you want to watch this with me? Tell me what you think as it goes."


class _Player:
    def __init__(self, opened_at: list[float]) -> None:
        opened_at.append(time.monotonic())
        self.states = iter([{"at": 2.0}, {"at": 5.0}, {"at": 8.0}, {"at": 11.0}, {"ended": True}, {"ended": True}])
        self.lines = iter([["the volcano wakes"], ["magma rises"], ["the island is born"], ["and it grows"]])

    async def says(self):
        return {"title": "Why volcanoes", "words": "A film about volcanoes."}

    async def start(self):
        return None

    async def state(self):
        return {"ended": False, "paused": False, "duration": 14.0, **next(self.states)}

    async def tend(self):
        return ""

    async def heard(self):
        return [{"words": w} for w in next(self.lines, [])]

    def eyes(self):
        async def look(_when):
            return "a volcano at sea"
        return look

    async def done(self):
        return None


def _a_person(monkeypatch):
    import core.skills.watching_along as watching_along
    import core.together.taking_it_up as taking_it_up
    from core.together.a_with import With

    held = With(who="person:bryan", what="watching Why volcanoes")
    taking = SimpleNamespace(her_said=[], left_by=[], _her_record=lambda: None,
                             _withs=lambda: SimpleNamespace(current=lambda _who: held, begin=lambda *a, **k: held))
    taking.she_said = lambda words, who, unprompted=False: taking.her_said.append((who, words, unprompted))
    taking.left = lambda who, by: (taking.left_by.append(by), held.leave(by, time.time()))
    monkeypatch.setattr(taking_it_up, "install", lambda: taking)
    monkeypatch.setattr(taking_it_up, "who_she_is_with", lambda _record: "person:bryan")
    monkeypatch.setattr(watching_along, "AN_END_HOLDS_S", 0.0)
    monkeypatch.setattr(watching_along, "into_her_senses", lambda *_: None)
    monkeypatch.setattr(watching_along, "kept_at", lambda *_a, **_k: (lambda _row: None))

    async def nothing(*_a, **_k):
        return None

    monkeypatch.setattr(watching_along, "into_her_memory", nothing)
    return held, taking, watching_along


def test_her_answer_goes_first_then_it_is_watched_together_and_the_with_ends_with_it(monkeypatch):
    import core.skills.watching_together as together

    held, taking, watching_along = _a_person(monkeypatch)
    monkeypatch.setattr(together, "owed_an_answer", lambda: False)
    observations, readings = [], []

    async def her_turn(request, observation, said):
        assert request == ASKED
        observations.append(observation)
        readings.append(together.what_we_are_watching())  # what a turn of the conversation would be handed now
        return f"thought {len(observations)}"

    async def say(_words):
        return True

    monkeypatch.setattr(watching_along, "her_turn", her_turn)
    monkeypatch.setattr(watching_along, "say_it", say)
    asking = SimpleNamespace(finalized=False)
    opened_at: list[float] = []

    async def go():
        async def answered():
            await asyncio.sleep(0.3)
            asking.settled_at, asking.finalized = time.monotonic(), True

        async def open_player():
            return _Player(opened_at)

        await asyncio.gather(answered(), together.watch_in_a_with(open_player, ASKED, title="Why volcanoes",
                                                                  url="https://videos.example/v/1", turn=asking,
                                                                  look_every_s=0.0, quiet_at_least_s=0.0))
    asyncio.run(go())

    assert opened_at[0] >= asking.settled_at  # her answer to the request went first
    assert not any(o.startswith("before it plays") for o in observations)  # her answer already said what she expects
    assert observations[-1].startswith("it has ended")
    assert any("WHAT IS PLAYING WHILE WE TALK" in r and "Why volcanoes" in r for r in readings)
    assert [w for _who, w, unprompted in taking.her_said if unprompted] == [f"thought {n}" for n in range(1, len(observations) + 1)]
    assert taking.left_by == ["it"] and not held.open  # over when the video is
    assert together.what_we_are_watching() == ""  # and no longer read into the conversation


def test_a_remark_waits_while_either_has_the_floor_or_a_turn_of_theirs_is_being_answered(monkeypatch):
    import core.skills.watching_together as together
    from core.together.a_with import HER, THEM, With

    held = With(who="person:bryan", what="watching")
    now = time.time()
    free = together.the_floor_is_free(held)
    monkeypatch.setattr(together, "owed_an_answer", lambda: False)
    assert free(now)
    held.said(THEM, now, until=now + 10)
    assert not free(now + 5)  # they are speaking
    held.said(HER, now + 20, until=now + 30)
    assert not free(now + 25)  # she is
    assert free(now + 40)
    monkeypatch.setattr(together, "owed_an_answer", lambda: True)
    assert not free(now + 40)  # her answer to them is being given: no remark across it


def test_a_turn_of_theirs_being_answered_is_read_from_her_orchestrator(monkeypatch):
    import core.skills.watching_together as together

    status = SimpleNamespace(is_processing=True)
    orchestrator = SimpleNamespace(status=status, _current_task_is_autonomous=False)
    monkeypatch.setattr("core.runtime.service_access.resolve_orchestrator", lambda **_k: orchestrator)
    assert together.owed_an_answer()
    orchestrator._current_task_is_autonomous = True
    assert not together.owed_an_answer()  # her own autonomous work is not somebody waiting
    status.is_processing, orchestrator._current_task_is_autonomous = False, False
    assert not together.owed_an_answer()
