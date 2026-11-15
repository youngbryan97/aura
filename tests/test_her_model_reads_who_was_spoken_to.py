"""Her address reader reads which talk heard words continue: her conversation, the room's, or none.

LIVE 13 November 2026 the phrase ladder this replaced answered a cooking show ("can you see how it thickens?") and a
phone call ("can you pick up some bread?") and stayed silent to "Hello. Is anyone there?". The reading is three
log-probabilities from a small reader model in a process of its own; these tests hold the arithmetic, the cache
handling and the process that produce them.
"""
from __future__ import annotations

import asyncio
import math
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.voice.duplex import address_reading
from core.voice.duplex.address_reading import (
    DOCUMENT_BEGINS,
    AddressMargin,
    AddressReader,
    AddressReading,
    RoomTalk,
    read_address,
)


def test_the_lead_is_over_the_better_of_the_room_and_nothing():
    assert AddressReading(hers=-1.0, alone=-4.0).lead == pytest.approx(3.0)
    assert AddressReading(hers=-1.0, alone=-4.0, room=-2.0).lead == pytest.approx(1.0)
    assert AddressReading(hers=-1.0, alone=-4.0, room=-2.0).rival == "the room's talk"
    assert AddressReading(hers=-1.0, alone=-2.0, room=-4.0).rival == "nothing at all"


def test_the_room_rivals_only_for_what_it_adds_over_talk_in_general():
    """LIVE 14 November 2026: after one overheard line, "Hello? Is anyone there?" read as the room's and went
    unanswered. Overheard lines lift any line of talk; the room is a rival only beyond that lift."""
    reading = AddressReading(hers=-2.19, alone=-4.80, room=-3.35, generic=-2.50)
    assert reading.room_beyond_talk == pytest.approx(-3.35 - (-2.50 + 4.80))
    assert reading.rival == "nothing at all" and reading.lead == pytest.approx(2.61)
    # A room that predicts the words beyond talk in general still wins.
    television = AddressReading(hers=-2.5, alone=-4.5, room=-1.0, generic=-3.5)
    assert television.rival == "the room's talk" and television.lead == pytest.approx(-0.5)
    assert "generic" in reading.as_dict()


def test_the_opening_is_read_only_when_the_room_has_talk_in_it():
    asked = []

    async def scorer(text, words, scope):
        asked.append(scope)
        return [-1.0]

    room = RoomTalk()
    asyncio.run(read_address("hi", hers_text="OPEN", hers_scope="h", room=room, room_scope="r", alone_scope="a",
                             scorer=scorer, opening="OPEN", opening_scope="o"))
    assert asked == ["h", "a"]
    room.overheard("the suspect was last seen heading north")
    asked.clear()
    reading = asyncio.run(read_address("hi", hers_text="OPEN", hers_scope="h", room=room, room_scope="r",
                                       alone_scope="a", scorer=scorer, opening="OPEN", opening_scope="o"))
    assert asked == ["h", "a", "r", "o"] and reading.generic == -1.0


def test_the_rooms_talk_keeps_its_latest_lines_within_its_bound():
    room = RoomTalk(keep_chars=30)
    assert room.text() is None
    for line in ["first line here", "second line here", "third line here"]:
        room.overheard(line)
    text = room.text()
    assert text.startswith(DOCUMENT_BEGINS) and text.endswith("third line here\n")
    assert "first line here" not in text
    room.overheard("   ")
    assert room.text() == text


def test_three_readings_each_the_mean_per_token_and_hers_read_first():
    asked = []

    async def scorer(text, words, scope):
        asked.append((text, words, scope))
        return {"hers-scope": [-1.0, -2.0, -3.0], "alone-scope": [-5.0, -5.0, -5.0],
                "room-scope": [-4.0, -4.0, -1.0]}[scope]

    room = RoomTalk()
    room.overheard("Previously, on the series.")
    reading = asyncio.run(read_address("  you   promised me ", hers_text="HER TEXT", hers_scope="hers-scope",
                                       room=room, room_scope="room-scope", alone_scope="alone-scope", scorer=scorer))
    assert [a[2] for a in asked] == ["hers-scope", "alone-scope", "room-scope"]
    assert all(a[1] == "you promised me" for a in asked)  # the words as her live context writes them
    assert asked[0][0] == "HER TEXT" and asked[1][0] == DOCUMENT_BEGINS and asked[2][0] == room.text()
    assert (reading.hers, reading.alone, reading.room, reading.tokens) == (-2.0, -5.0, -3.0, 3)


def test_nothing_overheard_means_no_room_reading_and_a_failed_read_means_no_reading():
    async def scorer(text, words, scope):
        return None if scope == "alone" else [-1.0]

    assert asyncio.run(read_address("hi", hers_text="T", hers_scope="h", room=RoomTalk(), room_scope="r",
                                    alone_scope="alone", scorer=scorer)) is None

    async def scorer_ok(text, words, scope):
        assert scope != "r"
        return [-1.0]

    reading = asyncio.run(read_address("hi", hers_text="T", hers_scope="h", room=RoomTalk(), room_scope="r",
                                       alone_scope="a", scorer=scorer_ok))
    assert reading.room is None


def test_a_margin_measured_on_another_model_is_not_used(monkeypatch):
    recorded = []
    monkeypatch.setattr("core.runtime.errors.record_degradation", lambda *a, **k: recorded.append(a))
    margin = AddressMargin(margin=1.5, model="Her-27B")
    reader = AddressReader(SimpleNamespace(scope="voice_live:s"), session_key="s", margin=margin,
                           model_name=lambda: "Another-27B")
    assert reader.margin is None and reader.margin is None
    assert len(recorded) == 1
    same = AddressReader(SimpleNamespace(scope="voice_live:s"), session_key="s", margin=margin,
                         model_name=lambda: "Her-27B")
    assert same.margin is margin


def test_the_margin_certificate_is_read_and_a_missing_one_is_none(tmp_path):
    assert AddressMargin.load(tmp_path / "absent.json") is None
    path = tmp_path / "address_margin.json"
    path.write_text('{"margin": 1.25, "model": "Her-27B", "measured_at": "2026-11-13"}')
    assert AddressMargin.load(path) == AddressMargin(margin=1.25, model="Her-27B", measured_at="2026-11-13")


def test_the_reader_reads_after_her_live_context_and_in_its_scope():
    asked = []

    class Live:
        scope = "voice_live:key"

        async def text_before(self, transcript):
            return "HER CONVERSATION<|im_start|>user\n"

    async def scorer(text, words, scope):
        asked.append((text, scope))
        return [-1.0]

    reader = AddressReader(Live(), session_key="key", scorer=scorer, margin=AddressMargin(1.0, "m"),
                           model_name=lambda: "m")
    reader.overheard("the suspect was last seen heading north")
    asyncio.run(reader.read("do you think he'll come back"))
    assert asked[0] == ("HER CONVERSATION<|im_start|>user\n", "hers:key")
    assert asked[1] == (DOCUMENT_BEGINS, "alone")
    assert asked[2][1] == "room:key"


# ── the reader's process ─────────────────────────────────────────────────

mx = pytest.importorskip("mlx.core")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "voice"))
import address_worker  # noqa: E402


class _Tokenizer:
    """One token per character."""

    @staticmethod
    def encode(text):
        return [ord(c) % 64 for c in text]


class _Cache:
    def __init__(self):
        self.offset = 0
        self.state = mx.array([0.0])


class _Model:
    """Logits that depend on each token's position and the token before it, so a cache trimmed to the wrong place
    gives different numbers. Records how many tokens it reads."""

    def __init__(self):
        self.read = []

    def __call__(self, ids, cache):
        tokens = [int(t) for t in ids[0].tolist()]
        self.read.append(len(tokens))
        rows = []
        for i, t in enumerate(tokens):
            position = cache[0].offset + i
            rows.append([((t * 7 + position * 3 + v) % 11) / 3.0 for v in range(64)])
        cache[0].offset += len(tokens)
        return mx.array([rows])


def _trim(cache, n):
    cache[0].offset -= n
    return n


def _reader(model):
    return address_worker.make_reader(model, _Tokenizer(), mx, lambda m: [_Cache()], _trim)


def test_a_conversation_that_grows_is_read_once_and_reads_the_same_as_reading_it_whole():
    model = _Model()
    read = _reader(model)
    conversation = "<|im_start|>user\nhello there<|im_end|>\n"
    first, count = read(conversation, "are you there", "hers")
    assert count == len(conversation) + len("are you there")
    grown = conversation + "<|im_start|>assistant\nyes<|im_end|>\n"
    second, count = read(grown, "good", "hers")
    assert count == len(grown) - len(conversation) + len("good")  # only what is new
    fresh, _ = _reader(_Model())(grown, "good", "elsewhere")
    assert second == pytest.approx(fresh)
    assert first == pytest.approx(_reader(_Model())(conversation, "are you there", "x")[0])


def test_other_words_after_the_same_text_trim_back_to_the_text():
    read = _reader(_Model())
    read("the room so far\n", "first line", "room")
    again, count = read("the room so far\n", "second line", "room")
    assert count == 1 + len("second line")
    assert again == pytest.approx(_reader(_Model())("the room so far\n", "second line", "r")[0])


def test_each_word_token_is_read_by_the_logits_before_it():
    model = _Model()
    values, _ = _reader(model)("ab", "cd", "s")
    tokens = _Tokenizer.encode("abcd")

    def expected(position, previous, token):
        row = [((previous * 7 + position * 3 + v) % 11) / 3.0 for v in range(64)]
        return row[token] - math.log(sum(math.exp(x) for x in row))

    assert values == pytest.approx([expected(1, tokens[1], tokens[2]), expected(2, tokens[2], tokens[3])], abs=1e-4)


FAKE = textwrap.dedent('''
    import json, sys, time
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request["words"] == "refuse":
            print(json.dumps({"error": "ValueError: nothing to read"}), flush=True)
        elif request["words"] == "late":
            time.sleep(5)
        elif request["words"] == "exit":
            sys.exit(0)
        else:
            print(json.dumps({"logprobs": [-1.0] * len(request["words"].split()), "read": 3}), flush=True)
''')


@pytest.fixture
def process(tmp_path):
    from core.voice.duplex.address_process import AddressProcess

    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE)
    running = AddressProcess("/models/Reader-1.5B", python=sys.executable, worker=worker)
    running.start(timeout_s=30)
    yield running
    running.close()


def test_a_reading_goes_to_the_process_and_comes_back(process):
    assert process.name == "Reader-1.5B"
    assert process.read("text", "two words", "hers") == [-1.0, -1.0]


def test_a_refused_reading_leaves_the_process_running(process):
    from core.voice.duplex.address_process import AddressReaderRefusedError

    with pytest.raises(AddressReaderRefusedError):
        process.read("text", "refuse", "hers")
    assert process.alive and process.read("text", "one", "hers") == [-1.0]


def test_a_late_or_lost_answer_closes_it_so_it_cannot_become_the_next_one(process):
    with pytest.raises(TimeoutError):
        process.read("text", "late", "hers", timeout_s=0.5)
    assert not process.alive
    with pytest.raises(RuntimeError, match="not running"):
        process.read("text", "one", "hers")


def test_a_process_that_exits_mid_reading_is_closed(process):
    with pytest.raises(RuntimeError, match="exited"):
        process.read("text", "exit", "hers")
    assert not process.alive
