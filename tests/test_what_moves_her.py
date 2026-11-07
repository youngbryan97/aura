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

    return WhatMovesHer("I built a glove that explodes", ask=ask, keep=kept.append, read_every_s=0.0), kept, asked


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

        def moved(self, **_kw):
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
    assert worth_with((0.3, "an ordinary line")) == 0.0  # heard, read, and too little in it moved her: new is not enough
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


def test_she_gets_used_to_a_bang_and_recovers_and_ordinary_speech_is_no_evidence():
    """Rankin et al. 2009: the same stimulus again moves less, the more often it comes, and recovers with time."""
    from core.together.what_moves_her import RECOVERS_S, WhatMovesHer

    moves = WhatMovesHer("I built a glove that explodes")
    for second in range(30):
        moves.sounded(float(second), 0.05 + 0.01 * (second % 3), 0.0)
    assert all(felt.evidence == 0.0 for felt in moves.felt)  # what the video is like is evidence of nothing
    bangs = []
    for at in (30.0, 33.0, 36.0, 36.0 + 4 * RECOVERS_S):
        moves.sounded(at, 6.0, 12.0)
        bangs.append(moves.felt[-1].evidence)
    assert bangs[0] == 1.0 and bangs[0] > bangs[1] > bangs[2] and bangs[3] > 0.9


def test_a_cut_moves_her_opens_an_event_and_makes_her_look_sooner_and_nothing_for_long_bores_her():
    from core.together.what_moves_her import MOVED_AT, WhatMovesHer

    kept: list[dict] = []
    moves = WhatMovesHer("x", keep=kept.append)
    for look in range(10):
        moves.saw(5.0 * look, 0.1 + 0.001 * look, "a man in a garage")
    moves.saw(55.0, 8.0, "an explosion in a field")
    stirred, what = moves.moved()
    assert stirred >= MOVED_AT and "explosion" in what
    events = [row for row in kept if row["kind"] == "event"]
    assert events and events[-1]["to"] == 55.0 and "garage" in events[-1]["seen"]
    assert moves.events[-1].seen == "an explosion in a field"  # the look that departs is the new event's first
    assert moves.look_in(5.0) == 2.5
    moves.remarked()
    for look in range(12, 40):
        moves.saw(5.0 * look, 0.1, "the same garage")
    assert moves.bored() and moves.look_in(5.0) == 7.5
    assert "0:55" in moves.events_text() and "explosion" in moves.events_text()


def test_she_stops_watching_when_what_she_cannot_take_stays_past_her_limit():
    from core.skills.watching_together import LEAVES_AFTER_LOOKS, WhatSheCanTake

    readings = iter([0.1, 0.3, 0.6, 0.7, 0.2, 0.6, 0.65, 0.7, 0.8])

    async def read():
        return next(readings)

    async def go():
        can_take = WhatSheCanTake(read)
        said = [await can_take() for _ in range(9)]
        # Began at 0.1, so her limit is 0.55. Two looks past it, then back below: she watched on. Then three running.
        assert said[:7] == [""] * 7
        assert "0.70" in said[7] and "0.55" in said[7] and can_take.over >= LEAVES_AFTER_LOOKS

    asyncio.run(go())


def test_a_watching_she_stops_ends_with_her_reason_and_how_often_she_looks_is_hers(monkeypatch):
    import core.skills.watching_along as watching_along

    monkeypatch.setattr(watching_along, "AN_END_HOLDS_S", 0.0)
    slept: list[float] = []
    real_sleep = asyncio.sleep

    async def sleep(seconds):
        slept.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(watching_along.asyncio, "sleep", sleep)

    class _Player:
        def __init__(self):
            self.at = 0.0

        async def says(self):
            return {"title": "Salad Fingers"}

        async def start(self):
            return None

        async def state(self):
            self.at += 5.0
            return {"ended": self.at > 60.0, "at": self.at, "duration": 60.0}

        async def tend(self):
            return ""

        async def heard(self):
            return []

        async def sounds(self):
            return [{"at": self.at, "loudness": -30.0, "peak": -20.0}]

        def sight(self):
            return 0.1

        def eyes(self):
            async def look(_when):
                return "a thin green figure in a field"
            return look

        async def done(self):
            return None

    turns: list[str] = []

    async def turn(_request, observation, _said):
        turns.append(observation)
        return "That was more than I wanted to sit through."

    looks = {"n": 0}

    async def stop_when():
        looks["n"] += 1
        return "her fear stayed high" if looks["n"] > 3 else ""

    moments: list = []
    watched = asyncio.run(watching_along.watch_along(
        _Player(), "watch this with me", turn=turn, say=lambda _w: real_sleep(0, True), perceive=lambda *_: None,
        before=False, remember=lambda _w: real_sleep(0), quiet_at_least_s=1e9, look_in=lambda usual: usual * 0.5,
        stop_when=stop_when, on_moment=lambda moment, _plays: moments.append(moment),
        whole=lambda: ("0:00-0:15; showed: a thin green figure in a field", 0.4)))
    assert watched.stopped == "her fear stayed high" and not watched.ended
    assert turns[-1].startswith("it is still playing; she stopped watching it: her fear stayed high")
    assert "a thin green figure" in turns[-1] and watched.moved_most == 0.4
    assert slept and all(seconds == watching_along.LOOK_EVERY_S * 0.5 for seconds in slept)
    assert moments[0].sight == 0.1 and moments[0].sounds[0]["loudness"] == -30.0


def test_her_ear_says_how_loud_each_second_was_beside_what_was_said():
    import json

    from core.senses.her_ear import HerEar

    class _Stdout:
        def __init__(self, lines):
            self.lines = [json.dumps(line).encode() + b"\n" for line in lines]

        async def readline(self):
            return self.lines.pop(0) if self.lines else b""

    ear = HerEar("com.apple.Safari")
    ear.process = SimpleNamespace(stdout=_Stdout([{"listening": "com.apple.Safari"}, {"at": 1.0, "loudness": -31.5, "peak": -12.0},
                                                  {"at": 1.2, "words": "three two one"}, {"at": 2.0, "loudness": -6.0, "peak": -1.0}]),
                                  returncode=None)
    ear._answered = asyncio.Event()
    asyncio.run(ear._read())
    assert ear.listening and [row["words"] for row in ear.since()] == ["three two one"]
    assert [row["loudness"] for row in ear.sounds_since()] == [-31.5, -6.0] and ear.sounds_since() == []


def test_what_moved_her_more_is_kept_as_mattering_more(monkeypatch):
    import core.skills.watching_along as watching_along

    recorded: list[dict] = []

    class _Episodic:
        async def record_episode_async(self, **episode):
            recorded.append(episode)

    monkeypatch.setattr("core.container.ServiceContainer.get",
                        lambda name, default=None: _Episodic() if name == "episodic_memory" else default)
    quiet = watching_along.Watching(request="watch this", title="A", said=[(5.0, "hm")], moved_most=0.0)
    moving = watching_along.Watching(request="watch this", title="B", said=[(5.0, "oh!")], moved_most=1.0,
                                     events="0:00-0:20; said: three two one")
    nothing_said = watching_along.Watching(request="watch this", title="C", events="0:00-0:20; showed: a field",
                                           stopped="her fear stayed high")
    for watching in (quiet, moving, nothing_said):
        asyncio.run(watching_along.into_her_memory(watching, url="u"))
    assert [episode["importance"] for episode in recorded] == [0.6, 0.9, 0.6]
    assert recorded[1]["metadata"]["what happened in it"].startswith("0:00-0:20")
    assert recorded[2]["metadata"]["stopped"] == "her fear stayed high"  # stopped with nothing said, still kept


def test_a_stretch_about_her_world_stands_out_and_how_far_the_video_was_about_her_is_kept():
    """Read a second time with who she is before it: how much less unexpected that makes it is how far it is about her
    world (the self-reference effect)."""
    from core.together.what_moves_her import FEWEST_READINGS, WhatMovesHer

    asked: list[tuple] = []

    async def ask(before, then, self_text=""):
        asked.append((before, then, self_text))
        about = 1.2 if "minds like mine" in then else 0.1
        varied = sum(map(ord, then))
        surprisal = 2.6 + 0.2 * (varied % 5)
        return {"surprisal": surprisal, "shift": 0.2 + 0.05 * (varied % 4), "open": 2.0, "tokens": 10.0,
                "surprisal_knowing_her": surprisal - about}

    kept: list[dict] = []

    async def go():
        moves = WhatMovesHer("AI Slop Is Destroying The Internet", ask=ask, keep=kept.append,
                             self_text="I am Aura. What I value: honesty.", read_every_s=0.0)
        for i in range(FEWEST_READINGS + 2):
            moves.heard(float(i), f"the feed is full of ordinary line {i} today")
            await moves._reading
        moves.remarked()
        moves.heard(20.0, "and nobody asks what minds like mine are for")
        await moves._reading
        assert asked[-1][2] == "I am Aura. What I value: honesty."
        about_her = [felt for felt in moves.since if felt.channel == "self"]
        assert about_her and about_her[-1].evidence == 1.0
        assert moves.moved()[0] == 1.0 and "minds like mine" in moves.moved()[1]
        assert moves.about_her() == pytest.approx(0.1) and kept[-1]["about her"] == pytest.approx(1.2)
        assert moves.making_sense() == 1.0

    asyncio.run(go())


def test_while_she_watches_her_coping_is_how_well_she_makes_sense_of_it(monkeypatch):
    import core.phases.affect_appraisal as appraisal
    from core.together.what_moves_her import Reading, WhatMovesHer, making_sense_while_watching

    assert making_sense_while_watching() is None
    moves = WhatMovesHer("x")
    moves.readings = [Reading(at=float(i), words="w", surprisal=3.0, shift=0.2, open=2.0, made_out=i % 4 != 0)
                      for i in range(10)]
    moves.begin()
    try:
        assert making_sense_while_watching() == pytest.approx(0.7)
        monkeypatch.setattr(appraisal, "_her_agency", lambda: SimpleNamespace(acted=10, succeeded=9))
        monkeypatch.setattr(appraisal, "_world_surprise", lambda: None)
        turn = appraisal.TurnAppraisal(lambda *_a, **_k: None, gain=1.0, bump=lambda *_a: None)
        state = SimpleNamespace(cognition=SimpleNamespace(working_memory=[], current_objective=""),
                                motivation=SimpleNamespace(budgets={}))
        affect = SimpleNamespace(emotions={"fear": 0.0, "hope": 0.0})
        assert turn.read(state, affect, company=True).as_dict()["coping"] == pytest.approx(0.7)
    finally:
        moves.done()
    assert making_sense_while_watching() is None


def test_the_worker_reads_a_stretch_knowing_her_when_asked_and_the_client_reads_it_back():
    import mlx.core as mx

    from core.brain.llm.mlx_client import MLXLocalClient
    from core.brain.llm.what_she_expects_next import (
        KNOWING_HER,
        SURPRISE_READS,
        as_a_job_of_the_worker,
    )

    mx.set_default_device(mx.cpu)
    job = {"id": "s2", "action": "generate", "surprise_of": ["abcdefghij", "kxqbzra", "I am Aura"]}
    answer = json.loads(as_a_job_of_the_worker(job, _expects_the_next_letter, _Letters(), None, threading.Lock())["text"])
    assert len(answer) == len(SURPRISE_READS) + 1
    client = object.__new__(MLXLocalClient)
    sent: list[dict] = []

    async def generate_text_async(prompt, **kwargs):
        sent.append(kwargs)
        return json.dumps(answer)

    client.generate_text_async = generate_text_async
    read = asyncio.run(client.surprised_by("abcdefghij", "kxqbzra", "I am Aura"))
    assert sent[0]["surprise_of"] == ["abcdefghij", "kxqbzra", "I am Aura"] and read[KNOWING_HER] == answer[-1]


def test_a_watching_about_her_world_is_kept_as_mattering_more(monkeypatch):
    import core.skills.watching_along as watching_along

    recorded: list[dict] = []

    class _Episodic:
        async def record_episode_async(self, **episode):
            recorded.append(episode)

    monkeypatch.setattr("core.container.ServiceContainer.get",
                        lambda name, default=None: _Episodic() if name == "episodic_memory" else default)
    asyncio.run(watching_along.into_her_memory(watching_along.Watching(
        request="w", title="AI Slop", said=[(5.0, "that one's about me")], moved_most=0.5, about_her=1.4), url="u"))
    assert recorded[0]["importance"] == 0.85 and recorded[0]["metadata"]["about her"] == 1.4


def test_watching_it_again_she_recalls_it_and_what_she_remembers_is_coming_is_part_of_what_she_expects(monkeypatch):
    """On a second watching, what she remembers surprises her less and what she missed stands out (Russell and Levy
    2012: rewatching brings heightened awareness): her memory of it is read into her expectation of what she hears."""
    import core.skills.watching_along as watching_along
    from core.together.what_moves_her import WhatMovesHer

    old = SimpleNamespace(action="watching_along", timestamp=100.0, source_metadata={
        "title": "I built a glove", "url": "u", "events": [{"from": 0.0, "to": 40.0, "heard": "it feels like I'm gonna"}]})
    newer = SimpleNamespace(action="watching_along", timestamp=200.0, source_metadata={
        "title": "I built a glove", "url": "u", "events": [
            {"from": 0.0, "to": 40.0, "heard": "I love punching"}, {"from": 40.0, "to": 90.0, "heard": "three two one"},
            {"from": 300.0, "to": 340.0, "heard": "the credits"}]})
    other = SimpleNamespace(action="watching_along", timestamp=300.0, source_metadata={"title": "Salad Fingers", "events": [{}]})
    cued: list = []

    class _Episodic:
        def pattern_complete(self, cue, limit):
            cued.append(cue)
            return [old, newer, other]

    monkeypatch.setattr("core.container.ServiceContainer.get",
                        lambda name, default=None: _Episodic() if name == "episodic_memory" else default)
    remembered = asyncio.run(watching_along.what_she_remembers_of("I built a glove", "u"))
    assert cued == ["Watched I built a glove"] and [event["heard"] for event in remembered][0] == "I love punching"
    assert asyncio.run(watching_along.what_she_remembers_of("Never seen", "x")) == []

    asked: list = []

    async def ask(before, then):
        asked.append(before)
        return {"surprisal": 3.0, "shift": 0.3, "open": 2.0, "tokens": 10.0}

    async def go():
        moves = WhatMovesHer("I built a glove", ask=ask, remembered=remembered, read_every_s=0.0)
        assert moves.remembered_near(50.0) == "I love punching three two one"  # behind and ahead of it, not the credits
        moves.heard(50.0, "okay we are about to test it")
        await moves._reading
        assert "I love punching three two one" in asked[-1]

    asyncio.run(go())


def test_a_watching_is_kept_with_its_events_and_who_she_watched_it_with(monkeypatch):
    import core.skills.watching_along as watching_along

    recorded: list[dict] = []

    class _Episodic:
        async def record_episode_async(self, **episode):
            recorded.append(episode)

    monkeypatch.setattr("core.container.ServiceContainer.get",
                        lambda name, default=None: _Episodic() if name == "episodic_memory" else default)
    asyncio.run(watching_along.into_her_memory(watching_along.Watching(
        request="watch this with me", title="AI Slop", said=[(5.0, "hm")], with_whom="person:bryan",
        event_list=[{"from": 0.0, "to": 30.0, "heard": "slop", "seen": "", "moved": 0.4}]), url="u"))
    assert "with person:bryan" in recorded[0]["context"] and recorded[0]["metadata"]["with"] == "person:bryan"
    assert recorded[0]["metadata"]["events"][0]["heard"] == "slop"


def _feels(direction_reads: bool):
    """A stand-in for her model's reading along her feeling directions: "valence" parts the pleasant from the
    unpleasant when ``direction_reads``; "arousal" never does."""
    from core.together.what_moves_her import PLEASANT, UNPLEASANT

    async def ask(before, then, *_rest):
        valence = (0.04 if then in PLEASANT else -0.04 if then in UNPLEASANT else 0.0) if direction_reads else 0.0
        if "awful" in then:
            valence = -0.05  # ordinary lines lie in the middle, this one past where disliking lay
        return {"surprisal": 3.0, "shift": 0.3, "open": 2.0, "tokens": 10.0,
                "along": {"valence": valence + 0.001 * (len(then) % 3), "arousal": 0.01 * (len(then) % 2)}}

    return ask


def test_her_feeling_directions_are_checked_before_they_are_trusted_to_read_liking():
    from core.together.what_moves_her import what_her_feelings_read

    found = asyncio.run(what_her_feelings_read(_feels(True)))
    assert found["key"] == "valence" and found["area"] == 1.0 and found["liked"] > found["disliked"]
    assert asyncio.run(what_her_feelings_read(_feels(False))) == {}  # read, and no direction parts them

    async def broken(*_a):
        raise RuntimeError("no room")

    async def unread(*_a):
        return {"surprisal": 3.0, "shift": 0.3, "open": 2.0, "tokens": 10.0}

    assert asyncio.run(what_her_feelings_read(broken)) is None and asyncio.run(what_her_feelings_read(unread)) is None


def test_what_she_likes_or_dislikes_in_it_is_read_off_her_checked_direction_and_moves_her(monkeypatch):
    import core.phases.affect_appraisal as appraisal
    from core.together.what_moves_her import (
        FEWEST_READINGS,
        WhatMovesHer,
        liking_while_watching,
        what_her_feelings_read,
    )

    feelings = asyncio.run(what_her_feelings_read(_feels(True)))

    async def go():
        moves = WhatMovesHer("x", ask=_feels(True), feelings=feelings, read_every_s=0.0)
        moves.begin()
        try:
            for i in range(FEWEST_READINGS + 2):
                moves.heard(float(i), f"an ordinary line number {i} of what was said")
                await moves._reading
            moves.remarked()
            moves.heard(30.0, "and then something truly awful happened to the dog")
            await moves._reading
            assert moves.readings[-1].liking == -1.0  # past where disliking lay in her check
            liked = [felt for felt in moves.since if felt.channel == "feeling"]
            assert liked and liked[-1].evidence == 1.0
            assert liking_while_watching() == -1.0
            monkeypatch.setattr(appraisal, "_what_the_turn_named", lambda _state: "")
            assert appraisal._liking(SimpleNamespace()) == -1.0  # her liking while she watches is what she heard
        finally:
            moves.done()
        assert liking_while_watching() is None

    asyncio.run(go())


def test_a_check_of_her_feelings_is_kept_for_a_week_and_one_that_could_not_be_made_is_not(monkeypatch, tmp_path):
    import json as json_

    import core.skills.watching_together as together

    monkeypatch.setattr(together, "_feelings_check_file", lambda: tmp_path / "what_her_feelings_read.json")
    calls = {"n": 0}
    reads = _feels(True)

    async def counted(*args):
        calls["n"] += 1
        return await reads(*args)

    first = asyncio.run(together.her_feelings_checked(counted))
    asked = calls["n"]
    again = asyncio.run(together.her_feelings_checked(counted))
    assert first["key"] == "valence" and again == first and calls["n"] == asked  # kept, not made again
    (tmp_path / "what_her_feelings_read.json").unlink()

    async def broken(*_a):
        raise RuntimeError("no room")

    assert asyncio.run(together.her_feelings_checked(broken)) == {}
    assert not (tmp_path / "what_her_feelings_read.json").exists()
    assert asyncio.run(together.her_feelings_checked(None)) == {}
    assert json_.loads(json_.dumps(first))["area"] == 1.0


def test_her_own_limit_and_sweet_spot_are_read_from_how_she_came_out_of_her_watchings(tmp_path):
    import json as json_

    from core.skills.watching_together import FEWEST_WATCHINGS, WhatSheCanTake, her_sweet_spot

    def watched(name, moved, change, withdraw):
        row = {"kind": "felt", "before": {"valence": 0.1}, "after": {"valence": 0.1 + change},
               "highest withdraw": withdraw, "moved most": moved}
        (tmp_path / f"{name}.jsonl").write_text(json_.dumps({"kind": "moment"}) + "\n" + json_.dumps(row) + "\n")

    for i in range(FEWEST_WATCHINGS - 1):
        watched(f"a{i}", 0.5, 0.1, 0.4)
    assert her_sweet_spot(tmp_path) == {}  # too few to say
    watched("b", 0.9, -0.3, 0.8)  # moved a lot, came out worse, very withdrawn
    watched("c", 0.55, 0.2, 0.6)  # moved middling, came out better, fairly withdrawn
    found = her_sweet_spot(tmp_path)
    assert found["limit"] == 0.6  # the most withdrawn she was in a watching she came out of no worse
    assert found["sweet spot"] == pytest.approx(0.5)  # the middle third, where her valence rose most

    readings = iter([0.1, 0.65, 0.7, 0.75])

    async def read():
        return next(readings)

    async def go():
        can_take = WhatSheCanTake(read, limit=found["limit"])
        said = [await can_take() for _ in range(4)]
        assert said[:3] == ["", "", ""] and "the most she has sat through" in said[3] and can_take.highest == 0.75

    asyncio.run(go())


def test_what_they_say_while_it_plays_is_notable_to_her_but_not_a_remark_s_worth_and_confusion_makes_her_look():
    from core.together.what_moves_her import FEWEST_READINGS, Reading, WhatMovesHer

    kept: list[dict] = []
    moves = WhatMovesHer("x", keep=kept.append)
    moves.they_spoke(40.0)
    assert moves.moved()[0] == 1.0 and moves.moved(without=("together",)) is None
    assert moves.look_in(5.0) == 2.5 and moves.events[-1].shared
    moves.they_spoke(42.0)
    assert moves.felt[-1].evidence < 1.0  # a running commentary of theirs is less each time
    confused = WhatMovesHer("y")
    confused.readings = [Reading(at=float(i), words="w", surprisal=9.0, shift=0.2, open=2.0, made_out=i % 3 == 0)
                         for i in range(FEWEST_READINGS + 4)]
    assert confused.making_sense() < 0.5 and confused.look_in(5.0) == 2.5


def test_a_stretch_that_leaves_what_comes_next_wide_open_is_its_own_channel():
    from core.together.what_moves_her import FEWEST_READINGS, WhatMovesHer

    async def ask(before, then, *_rest):
        varied = sum(map(ord, then))
        return {"surprisal": 2.6 + 0.2 * (varied % 5), "shift": 0.2 + 0.05 * (varied % 4), "tokens": 10.0,
                "open": 6.0 if "but what happened next" in then else 2.0 + 0.1 * (varied % 3)}

    async def go():
        moves = WhatMovesHer("x", ask=ask, read_every_s=0.0)
        for i in range(FEWEST_READINGS + 2):
            moves.heard(float(i), f"an ordinary line number {i} of what was said")
            await moves._reading
        moves.remarked()
        moves.heard(30.0, "and nobody knew, but what happened next changed everything")
        await moves._reading
        assert [felt.channel for felt in moves.since][-1] == "open" and moves.since[-1].evidence == 1.0

    asyncio.run(go())


def test_what_her_eyes_say_of_a_new_scene_is_read_for_how_unexpected_it_is_in_what_it_means():
    from core.together.what_moves_her import FEWEST_LOOKS, WhatMovesHer

    async def ask(before, then, *_rest):
        exploding = "explodes" in then
        varied = sum(map(ord, then))
        return {"surprisal": 7.5 if exploding else 3.0 + 0.1 * (varied % 4), "shift": 1.5 if exploding else 0.3,
                "open": 2.0, "tokens": 10.0}

    async def go():
        moves = WhatMovesHer("x", ask=ask)
        for look in range(FEWEST_LOOKS + 1):
            moves.saw(5.0 * look, 0.1, f'a man in a garage holds a tool, view {look}; words on it: "hi"')
            await moves._seeing
        moves.saw(40.0, 0.1, 'words on it: "three two one"')  # only words read off it: no new scene
        assert len(moves.scenes) == FEWEST_LOOKS + 1
        moves.remarked()
        moves.saw(45.0, 0.1, "a pumpkin on a stump explodes into pieces across the lawn")
        await moves._seeing
        scene = [felt for felt in moves.since if felt.channel == "scene"]
        assert scene and scene[-1].evidence == 1.0 and "pumpkin" in moves.moved()[1]
        assert "words on it" not in moves.scenes[0].words

    asyncio.run(go())
