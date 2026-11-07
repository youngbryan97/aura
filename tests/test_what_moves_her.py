"""What moves her in what she watches, read off her own expectations of what she hears (core/together/what_moves_her.py),
and what a remark is worth by it (core/skills/watching_together.py WorthSaying). Her model is a stand-in here: in her
it is her cortex, read by its worker with nothing sampled (core/brain/llm/what_she_expects_next.py surprise_of)."""
from __future__ import annotations

import asyncio
import json
import math
import threading
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

V = 50


class _Letters:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) % V for char in str(text)]


def _expects_the_next_letter(batch):
    """A stand-in that expects each letter to be followed by the next one."""
    import mlx.core as mx

    following = (batch + 1) % V
    return mx.where(mx.arange(V)[None, None, :] == following[..., None], 8.0, 0.0).astype(mx.float32)


def test_what_she_heard_is_read_for_how_unexpected_it_was_how_far_it_moved_her_and_how_open_it_leaves_her():
    import mlx.core as mx

    from core.brain.llm.what_she_expects_next import (
        SURPRISE_READS,
        as_a_job_of_the_worker,
        surprise_of,
    )

    mx.set_default_device(mx.cpu)
    expected = surprise_of(_expects_the_next_letter, _Letters(), "abcdefghij", "klmnopq")
    broken = surprise_of(_expects_the_next_letter, _Letters(), "abcdefghij", "kxqbzra")
    assert set(expected) == set(SURPRISE_READS) and all(math.isfinite(value) for value in broken.values())
    assert expected["surprisal"] < broken["surprisal"] and expected["tokens"] == 8.0  # its leading space and 7 letters
    assert surprise_of(_expects_the_next_letter, _Letters(), "a", "klm") == {}  # too little before it to read against
    steering = SimpleNamespace(alpha=5.4, during=[], set_alpha=None, controlled_measurement=None)

    import contextlib

    @contextlib.contextmanager
    def held():
        before = steering.alpha
        try:
            yield
        finally:
            steering.alpha = before

    steering.controlled_measurement = held
    steering.set_alpha = lambda alpha: (steering.during.append(alpha), setattr(steering, "alpha", alpha))
    job = {"id": "s1", "action": "generate", "surprise_of": ["abcdefghij", "kxqbzra"]}
    answer = as_a_job_of_the_worker(job, _expects_the_next_letter, _Letters(), steering, threading.Lock())
    # A list in the order of SURPRISE_READS, the shape the other pass answers in, unsteered and back.
    assert json.loads(answer["text"]) == pytest.approx([broken[key] for key in SURPRISE_READS])
    assert answer["status"] == "ok" and steering.during == [0.0] and steering.alpha == 5.4
    assert json.loads(as_a_job_of_the_worker({"id": "s", "surprise_of": ["one"]}, _expects_the_next_letter, _Letters(),
                                             None, threading.Lock())["text"]) == []


def test_the_client_asks_its_worker_once_and_reads_back_only_a_whole_reading():
    from core.brain.llm.mlx_client import MLXLocalClient

    client = object.__new__(MLXLocalClient)
    sent: list[dict] = []

    async def generate_text_async(prompt, **kwargs):
        sent.append({"prompt": prompt, **kwargs})
        return json.dumps([3.25, 0.5, 1.75, 12.0])

    client.generate_text_async = generate_text_async
    assert asyncio.run(client.surprised_by("before it", "what was heard")) == {
        "surprisal": 3.25, "shift": 0.5, "open": 1.75, "tokens": 12.0}
    assert sent[0]["surprise_of"] == ["before it", "what was heard"] and sent[0]["internal_inference"] is True
    assert sent[0]["max_tokens"] == 1 and sent[0]["temperature"] == 0.0 and sent[0]["foreground_request"] is True

    async def half(prompt, **kwargs):
        return json.dumps([3.25, 0.5])

    client.generate_text_async = half
    assert asyncio.run(client.surprised_by("before it", "what was heard")) == {}


def _heard_with(readings: dict[str, tuple[float, float]]):
    """A stand-in for her model: each stretch's (surprisal, shift) by a word in it."""
    from core.together.what_moves_her import WhatMovesHer

    kept: list[dict] = []
    asked: list[tuple[str, str]] = []

    async def ask(before: str, then: str) -> dict[str, float]:
        asked.append((before, then))
        for word, (surprisal, shift) in readings.items():
            if word in then.lower():
                return {"surprisal": surprisal, "shift": shift, "open": 2.0, "tokens": 10.0}
        varied = sum(map(ord, then))  # ordinary speech, read a little differently each time
        return {"surprisal": 2.6 + 0.2 * (varied % 5), "shift": 0.2 + 0.05 * (varied % 4), "open": 2.0, "tokens": 10.0}

    return WhatMovesHer("I built a glove that explodes", ask=ask, keep=kept.append), kept, asked


def test_a_stretch_is_read_once_it_is_a_stretch_and_one_at_a_time():
    async def go():
        moves, kept, asked = _heard_with({})
        moves.heard(1.0, "he picks up")  # three words: not yet a stretch
        assert not asked and moves._reading is None
        moves.heard(2.0, "he picks up")  # the same words heard again are not new
        moves.heard(3.0, "the gauntlet and grins")
        await moves._reading
        assert asked[-1] == ("I built a glove that explodes", "he picks up the gauntlet and grins")
        moves.heard(4.0, "a pumpkin sits on the stump waiting")
        moves.heard(5.0, "three two one go go go")  # a stretch, but that one is still being read: it waits
        await moves._reading
        moves.heard(6.0, "and it goes")
        await moves._reading
        assert asked[-1][1] == "three two one go go go and it goes"
        assert "he picks up the gauntlet and grins" in asked[-1][0]  # what came before it is what it is read against
        assert [row["kind"] for row in kept] == ["moved"] * 3

    asyncio.run(go())


def test_what_moves_her_is_what_stands_out_in_this_video_and_a_line_she_could_not_make_out_moves_nothing():
    from core.together.what_moves_her import FEWEST_READINGS

    async def go():
        moves, kept, _asked = _heard_with({"thirty feet": (6.0, 1.4), "zxqv": (40.0, 3.0)})
        for i in range(FEWEST_READINGS + 2):
            moves.heard(float(i), f"ordinary line number {i} about the glove")
            await moves._reading
        assert moves.moved() is not None and moves.moved()[0] < 0.8  # ordinary for this video
        moves.remarked()
        assert moves.moved() is None  # nothing heard since she spoke
        moves.heard(20.0, "zxqv zxqv blorp grnn shnk wuu")  # a misheard line
        await moves._reading
        assert kept[-1]["made_out"] is False and moves.moved() == (0.0, "zxqv zxqv blorp grnn shnk wuu")
        moves.heard(21.0, "THAT WENT LIKE THIRTY FEET oh my god")
        await moves._reading
        stirred, words = moves.moved()
        assert stirred >= 0.8 and "THIRTY FEET" in words
        moves.done()
        assert kept[-1]["kind"] == "moved overall" and kept[-1]["not made out"] == 1

    asyncio.run(go())


def test_a_remark_is_worth_what_moved_her_and_newness_only_where_nothing_she_heard_was_read(monkeypatch):
    from core.skills.watching_together import NEW_AND_UNKNOWN, WorthSaying
    from core.together import remarks
    from core.together import when_to_say_something as rule
    from core.together.a_with import With

    monkeypatch.setattr(remarks, "interest_in", lambda _seen, _change: (0.0, ""))
    weighed: list[float] = []
    monkeypatch.setattr(rule, "worth_saying_now",
                        lambda held, now, *, interest, thing, lost_by_waiting: weighed.append(interest) or SimpleNamespace(
                            speak=interest > 0, why=""))

    class Moves:
        def __init__(self, moved):
            self.value, self.cleared = moved, 0

        def moved(self):
            return self.value

        def remarked(self):
            self.cleared += 1

    def worth_with(moved) -> float:
        moves = Moves(moved)
        worth = WorthSaying(With(who="person:bryan", what="watching a video", began=0.0), moves=moves)
        worth.took_in(SimpleNamespace(at=0.0, seen="a man in a garage holds a glove", read=""))
        worth(None, 1.0)
        worth.remarked()
        assert moves.cleared == 1
        return weighed[-1]

    assert worth_with((0.93, "THAT WENT LIKE THIRTY FEET")) == pytest.approx(0.93)
    assert worth_with((0.55, "an ordinary line")) == 0.0  # heard, read, and nothing in it moved her: new is not enough
    assert worth_with(None) == NEW_AND_UNKNOWN  # nothing she heard was read: what is new to her stands in


def test_a_remark_of_hers_is_felt_for_what_moved_her_while_she_watches(monkeypatch):
    """Her remarks were felt for her world model's surprise at her own state alone: what she had just heard, which moved
    her to speak, reached none of her feelings. While she watches, her appraisal's novelty is the greater of the two."""
    import core.phases.affect_appraisal as appraisal
    from core.together.what_moves_her import WhatMovesHer, moved_while_watching

    monkeypatch.setattr(appraisal, "_world_surprise", lambda: 0.3)
    assert moved_while_watching() is None and appraisal._novelty() == 0.3  # watching nothing: her world model's
    moves = WhatMovesHer("I built a glove that explodes")
    monkeypatch.setattr(moves, "moved", lambda: (0.92, "THAT WENT LIKE THIRTY FEET"))
    moves.begin()
    try:
        assert moved_while_watching() == 0.92 and appraisal._novelty() == 0.92
        monkeypatch.setattr(moves, "moved", lambda: (0.1, "an ordinary line"))
        assert appraisal._novelty() == 0.3  # her world model surprised her more
    finally:
        moves.done()
    assert moved_while_watching() is None and appraisal._novelty() == 0.3
    monkeypatch.setattr(appraisal, "_world_surprise", lambda: None)
    assert appraisal._novelty() is None
