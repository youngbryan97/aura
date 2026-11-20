"""A question about the world that names a language is not work on code (Master Todo U07).

LIVE 19 November 2026, four times: "Can you look up what the latest stable release of Python is right now?" was read
as code by a word match on "Python", and a deep search of nine candidates ran over her answer until the turn was out of
time. The kind is now read by meaning on her encoder, which may veto the word match's "code" and nothing else.
"""
from __future__ import annotations

import pytest

from core.brain import reasoning_amplifier_v2 as amplifier
from core.intent import asks_for_code_work as reader

LIVE = "Can you look up what the latest stable release of Python is right now?"
#: Held out of the declaration: none of these is among its examples.
HELD_OUT_CODE_WORK = [
    "Can you write a Python script that renames every file in a folder?",
    "My Flask route returns a 500 error, here is the traceback, what is wrong?",
    "Convert this loop into a list comprehension.",
    "Can you review the code in my pull request and point out bugs?",
]
HELD_OUT_ABOUT_THE_WORLD = [
    LIVE,
    "What year did Guido van Rossum step down as Python's leader?",
    "Is Kotlin used more than Java on Android today?",
    # LIVE 19 November 2026, evening, after a restart: read as work on code by the word "code".
    "What is the locker code for the Brimvaltor depot?",
    "What's the zip code for Beverly Hills?",
    "What is the gate code for the Telgordal depot?",
]


class _Surface:
    def __init__(self, says):
        self.says = says

    def decide(self, text):
        return self.says


@pytest.fixture
def surface(monkeypatch):
    def set_to(says):
        monkeypatch.setattr(reader, "_surface", lambda: _Surface(says))
    return set_to


def test_a_request_the_surface_reads_as_about_the_world_is_not_amplified_as_code(surface):
    surface(False)
    assert amplifier.is_amplifiable(LIVE) is None


@pytest.mark.parametrize("says", [True, None])
def test_where_the_surface_says_code_or_cannot_tell_the_words_stand(surface, says):
    surface(says)
    assert amplifier.is_amplifiable(LIVE) == "code"


def test_the_surface_reaches_only_a_code_reading(surface, monkeypatch):
    surface(False)
    assert amplifier.is_amplifiable("What is 17 * 23?") == "math", "a computation is not vetoed by it"


def test_on_an_event_loop_it_never_computes(monkeypatch):
    import asyncio

    asked: list[str] = []

    class _Counting:
        def decide(self, text):
            asked.append("decide")
            return False

        def decide_without_waiting(self, text):
            asked.append("without waiting")
            return None

    monkeypatch.setattr(reader, "_surface", lambda: _Counting())

    async def on_her_loop():
        return reader.asks_for_code_work(LIVE)

    monkeypatch.setattr(reader, "_DECIDED", {})
    assert asyncio.run(on_her_loop()) is None and asked == ["without waiting"]
    assert reader.asks_for_code_work(LIVE) is False and asked[-1] == "decide", "off the loop it decides"
    assert asyncio.run(on_her_loop()) is False and asked[-1] == "decide", "and on the loop it reads what was decided"


def test_with_no_encoder_the_reader_says_nothing(monkeypatch):
    monkeypatch.setattr(reader, "_surface", lambda: None)
    assert reader.asks_for_code_work(LIVE) is None


def test_no_held_out_request_is_among_the_declared_examples():
    declared = {*reader._CODE_WORK, *reader._ABOUT_THE_WORLD}
    assert not declared & {*HELD_OUT_CODE_WORK, *HELD_OUT_ABOUT_THE_WORLD}


@pytest.fixture
def her_encoder():
    found = reader._surface()
    if found is None or not found._prepare():
        pytest.skip("her encoder is not available here")
    return found


@pytest.mark.model
@pytest.mark.parametrize("request_text", HELD_OUT_ABOUT_THE_WORLD)
def test_on_her_encoder_a_question_about_the_world_is_not_code_work(her_encoder, request_text):
    assert her_encoder.decide(request_text) is not True, request_text


@pytest.mark.model
@pytest.mark.parametrize("request_text", HELD_OUT_CODE_WORK)
def test_on_her_encoder_work_on_code_is_never_vetoed(her_encoder, request_text):
    assert her_encoder.decide(request_text) is not False, request_text


@pytest.mark.model
def test_on_her_encoder_the_live_question_is_read_as_about_the_world(her_encoder):
    assert her_encoder.decide(LIVE) is False
