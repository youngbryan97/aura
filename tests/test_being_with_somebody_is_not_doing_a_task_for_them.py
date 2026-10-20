"""Being with somebody is told from doing a task for them, and is lived differently.

Bryan, 13 October 2026: "I can ask Aura 'hey, watch this youtube video for 10
minutes and comment on it as you watch.' and she'll do that. as a task. but
that's different from asking 'Hey do you want to watch this video with me?'
and her either saying 'Sure!' or 'What video? May be interested.' Or 'Nah, I'm
ok for now.' ... all as a way of *being*."

The package under test is core/together.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import pytest

from core.conation.engine import ConationEngine
from core.conation.what_she_wants import WhatSheWants
from core.together.a_with import HER, IT, THEM, Together, With
from core.together.her_answer import her_answer
from core.together.taking_it_up import HEADER, TakingItUp
from core.together.what_was_proposed import KINDS, SECONDS, WhatItAsksOfHer
from core.together.when_to_say_something import worth_saying_now

INES = "person:ines"
FILM, HERON = "topic:film", "topic:heron"


# ── a with ──────────────────────────────────────────────────────────


def _talked(held: With, gaps=(4.0, 6.0, 5.0, 30.0, 8.0), start: float = 100.0) -> float:
    """They and she take turns, one second each, leaving these silences. Returns when the last ended."""
    at = start
    for index, gap in enumerate([0.0, *gaps]):
        at += gap
        held.said(THEM if index % 2 == 0 else HER, at, until=at + 1.0)
        at += 1.0
    return at


def test_a_with_reads_its_own_silences_and_has_no_other_clock():
    held = With(who=INES, what=FILM, began=100.0)
    assert held.moment(101.0) is None  # no silence left yet, nothing to read
    ended = _talked(held)
    assert list(held.silences) == [4.0, 6.0, 5.0, 30.0, 8.0]
    assert held.moment(ended + 1.0) == 0.0
    assert held.moment(ended + 5.5) == pytest.approx(2 / 5)
    assert held.moment(ended + 40.0) == 1.0


def test_while_somebody_is_speaking_it_is_nobodys_moment():
    held = With(who=INES, what=FILM, began=0.0)
    ended = _talked(held)
    held.said(THEM, ended + 10.0)
    assert held.speaking(ended + 11.0) == THEM
    assert held.moment(ended + 11.0) == 0.0
    held.stopped_speaking(THEM, ended + 14.0)
    assert held.speaking(ended + 15.0) == ""
    assert held.quiet(ended + 15.0) == pytest.approx(1.0)


def test_what_one_presents_is_grounded_when_the_other_names_it_or_goes_on():
    held = With(who=INES, what=FILM, began=0.0)
    held.said(THEM, 1.0, about=[HERON, FILM], until=2.0)
    assert not held.in_common(HERON)  # presented, and nothing yet shows it was taken in
    held.said(HER, 3.0, about=[HERON], until=4.0)
    assert held.ground[HERON].evidence == "named"
    assert held.ground[FILM].evidence == "went on"
    assert held.in_common(HERON) and held.in_common(FILM)


def test_what_is_in_view_of_both_is_common_ground_without_being_said():
    held = With(who=INES, what=FILM, began=0.0)
    held.shown("a boat on a lake", 5.0, change=0.2)
    held.shown("a kitchen at night", 9.0, change=0.9)
    held.shown("the same kitchen", 12.0, change=0.1)
    assert held.in_common("a kitchen at night")
    assert held.in_view(10.0).what == "a kitchen at night"
    # "Go back": what was showing before what is showing now.
    assert held.before(13.0).what == "a kitchen at night"
    assert held.before(13.0, steps=2).what == "a boat on a lake"
    assert held.before(6.0) is None


def test_her_remarks_are_counted_by_whether_they_were_taken_up():
    held = With(who=INES, what=FILM, began=0.0)
    ended = _talked(held)
    assert held.welcome() == pytest.approx(1 / 2)  # nothing known of how she is taken
    held.said(HER, ended + 20.0, about=[HERON], until=ended + 21.0, unprompted=True)
    held.said(THEM, ended + 22.0, about=[HERON], until=ended + 23.0)  # named it back
    assert (held.remarks, held.taken_up) == (1, 1)
    held.said(HER, ended + 40.0, about=[FILM], until=ended + 41.0, unprompted=True)
    held.said(HER, ended + 80.0, about=[FILM], until=ended + 81.0, unprompted=True)  # spoke again first
    held.said(THEM, ended + 200.0, until=ended + 201.0)  # much later than they usually answer
    assert (held.remarks, held.taken_up, held.went_by) == (3, 1, 1)
    # The one she spoke past went by, neither taken up nor turned down (LIVE 20 October 2026, a watching in silence);
    # the one answered much later than they usually answer was not taken up.
    assert held.welcome() == pytest.approx(2 / 4)


def test_a_with_is_ended_by_leaving_and_by_nothing_else():
    withs = Together()
    held = withs.begin(INES, FILM, at=0.0)
    assert withs.begin(INES, FILM, at=5.0) is held  # proposing it again is the same with
    _talked(held)
    assert withs.current(INES) is held and held.open
    ended = withs.end(INES, by=THEM, at=900.0)
    assert ended is held and not held.open and held.left_by == THEM
    assert withs.current(INES) is None and withs.current() is None


# ── whether to speak inside one ─────────────────────────────────────


def _watching(welcomed: int = 0, passed: int = 0) -> tuple[With, float]:
    held = With(who=INES, what=FILM, began=0.0)
    ended = _talked(held)
    held.remarks, held.taken_up = welcomed + passed, welcomed
    for index, change in enumerate((0.2, 0.9, 0.5, 0.1, 0.5)):
        held.shown(f"scene {index}", ended - 50.0 + 10 * index, change=change)
    return held, ended


def test_spoken_to_she_answers_once_the_floor_is_free():
    held, ended = _watching()
    assert worth_saying_now(held, ended + 0.2, interest=0.0, spoken_to=True).speak
    held.said(THEM, ended + 5.0)
    assert not worth_saying_now(held, ended + 6.0, interest=1.0, spoken_to=True).speak


def test_she_does_not_talk_over_them_and_does_not_say_a_thing_twice():
    held, ended = _watching(welcomed=6)
    held.said(THEM, ended + 5.0)
    assert worth_saying_now(held, ended + 6.0, interest=1.0, thing=HERON).why == "they are speaking"
    held.stopped_speaking(THEM, ended + 7.0)
    held.said(HER, ended + 30.0, about=[HERON], until=ended + 31.0, unprompted=True)
    said = worth_saying_now(held, ended + 90.0, interest=1.0, thing=HERON)
    assert not said.speak and said.why == "she has said her piece about it"


def test_the_same_remark_waits_for_a_lull_and_is_made_in_one():
    held, ended = _watching(welcomed=6)
    early = worth_saying_now(held, ended + 1.0, interest=0.8, thing=HERON)
    later = worth_saying_now(held, ended + 40.0, interest=0.8, thing=HERON)
    assert not early.speak and later.speak
    assert later.welcome > early.welcome


def test_a_thing_of_little_interest_is_not_worth_his_attention_and_one_of_much_is():
    held, ended = _watching(welcomed=2, passed=2)
    assert not worth_saying_now(held, ended + 7.0, interest=0.1, thing=HERON).speak
    assert worth_saying_now(held, ended + 7.0, interest=0.9, thing=HERON, lost_by_waiting=1.0).speak


def test_she_grows_quiet_with_somebody_who_lets_her_remarks_go_by():
    taken, ended = _watching(welcomed=8)
    passed, _ended = _watching(passed=8)
    # The same lull, the same thing, the same interest in it.
    assert worth_saying_now(taken, ended + 40.0, interest=0.6, thing=HERON).speak
    assert not worth_saying_now(passed, ended + 40.0, interest=0.6, thing=HERON).speak


def test_what_the_decision_weighed_can_be_read_back():
    held, ended = _watching(welcomed=3, passed=1)
    said = worth_saying_now(held, ended + 7.0, interest=0.5, thing=HERON)
    assert said.for_speaking == pytest.approx(said.welcome * 0.5)
    assert said.against == pytest.approx((1.0 - said.welcome) * said.cost)
    assert said.speak == (said.for_speaking > said.against)


# ── what an utterance asks of her ───────────────────────────────────

USUAL = {second: -6.0 - 0.3 * index for index, second in enumerate(s for kind in KINDS for s in SECONDS[kind])}


def _a_model_that_expects(kind_of: dict[str, str], lift: float = 3.0):
    """Scores each second part at its usual likelihood, and higher after an utterance of its kind."""

    def scorer(said, continuations):
        kind = kind_of.get(said, "")
        return [USUAL[c] + (lift if kind and c in SECONDS[kind] else 0.0) for c in continuations]

    return scorer


SAID = {
    "Do you want to watch this video with me?": "join",
    "Watch this video for ten minutes and comment on it.": "do",
    "Why does a river wind?": "ask",
    "The wrens left the boot this morning.": "tell",
    "Good morning.": "greet",
}


def _reader(heard_before: int = 12) -> WhatItAsksOfHer:
    reader = WhatItAsksOfHer(_a_model_that_expects(SAID))
    for index in range(heard_before):
        reader.remember(f"something somebody once said, number {index}")
    return reader


def test_an_utterance_is_the_kind_whose_answers_became_likelier_after_it():
    reader = _reader()
    for said, kind in SAID.items():
        read = reader.heard(said)
        assert read.kind == kind and read.sure > 0.5, said
        assert sum(read.by_kind.values()) == pytest.approx(1.0)


def test_an_answer_that_is_merely_common_does_not_win():
    """The first second part scores highest after everything, and reads as nothing."""
    reader = _reader()
    read = reader.heard("a thing that asks nothing in particular")
    assert max(read.by_kind.values()) == pytest.approx(1 / len(KINDS), abs=0.02)


def test_with_nothing_to_set_it_against_nothing_is_read():
    reader = WhatItAsksOfHer(_a_model_that_expects(SAID))
    read = reader.heard("Do you want to watch this video with me?")
    assert read.kind == "" and read.heard_before == 0
    assert reader.heard("Do you want to watch this video with me?").heard_before == 1


def test_a_model_that_could_not_score_reads_nothing_and_teaches_nothing():
    reader = WhatItAsksOfHer(lambda said, continuations: [])
    assert reader.heard("Do you want to watch this with me?").kind == ""
    assert reader.heard_before() == 0
    nan = WhatItAsksOfHer(lambda said, continuations: [math.nan] * len(continuations))
    assert not nan.remember("anything")


def test_what_the_answers_usually_score_is_kept():
    reader = _reader()
    again = WhatItAsksOfHer(_a_model_that_expects(SAID), reader.held())
    assert again.heard_before() == 12
    assert again.heard("Do you want to watch this video with me?").kind == "join"


def test_no_second_part_belongs_to_two_kinds():
    seen: dict[str, str] = {}
    for kind in KINDS:
        for second in SECONDS[kind]:
            assert second not in seen, f"{second!r} is offered for {seen.get(second)} and {kind}"
            seen[second] = kind


# ── her answer ──────────────────────────────────────────────────────


@pytest.fixture
def record(tmp_path):
    return WhatSheWants(tmp_path / "wants.json", engine=ConationEngine())


def _lived(record, paid: dict[str, float], turns: int = 20) -> None:
    for _ in range(turns):
        for key, worth in paid.items():
            record.met([key], worth, reachable=[key], labels={key: key.split(":", 1)[1]})


def test_she_joins_what_her_record_predicts_is_worth_more_than_staying(record):
    _lived(record, {FILM: 0.6, INES: 0.3})
    answer = her_answer(record, INES, [FILM])
    assert answer.joins is True
    assert answer.worth_of_joining > 0.5 > answer.worth_of_staying


def test_she_says_not_now_when_what_she_is_in_the_middle_of_is_worth_more(record):
    _lived(record, {FILM: 0.1, HERON: 0.2})
    record.started(HERON, "reading about herons", label="heron")
    for _ in range(12):
        record.met([HERON], 0.8, reachable=[HERON], labels={HERON: "heron"})
    answer = her_answer(record, INES, [FILM])
    assert answer.joins is False
    assert answer.leaving == HERON and answer.worth_of_staying > answer.worth_of_joining


def test_she_says_not_now_to_what_she_has_found_she_dislikes(record):
    for turn in range(40):
        key, worth = ((FILM, -0.6), (HERON, 0.6))[turn % 2]
        record.met([key], worth, reachable=[key], labels={key: key.split(":", 1)[1]})
    answer = her_answer(record, INES, [FILM])
    assert answer.joins is False and "dislikes film" in answer.why


def test_she_asks_what_it_is_when_nothing_she_has_met_settles_it(record):
    answer = her_answer(record, INES, [FILM])
    assert answer.joins is None
    assert answer.unknown == (FILM,)
    assert "is new to her" in answer.why


def test_somebody_whose_company_has_paid_is_reason_enough(record):
    _lived(record, {INES: 0.5})
    answer = her_answer(record, INES, [FILM])
    assert answer.joins is True and answer.unknown == (FILM,)


# ── taking it up ────────────────────────────────────────────────────


def _taking(record) -> tuple[TakingItUp, Together]:
    withs = Together()
    return TakingItUp(_reader(), record=record, together=withs), withs


def test_an_invitation_she_wants_begins_a_with_and_a_request_does_not(record):
    _lived(record, {INES: 0.5})
    taking, withs = _taking(record)
    asked = taking.read("Watch this video for ten minutes and comment on it.", who=INES, at=10.0)
    assert asked["proposal"]["kind"] == "do" and asked["answer"] is None
    assert withs.current(INES) is None
    invited = taking.read("Do you want to watch this video with me?", who=INES, at=20.0)
    assert invited["proposal"]["kind"] == "join" and invited["answer"]["joins"] is True
    held = withs.current(INES)
    assert held is not None and held.proposed_by == THEM
    assert record.thing(INES).under_way  # the doing is lived turn by turn from here


def test_an_invitation_she_does_not_want_begins_nothing(record):
    _lived(record, {HERON: 0.2})
    record.started(HERON, "reading about herons", label="heron")
    for _ in range(12):
        record.met([HERON], 0.8, reachable=[HERON], labels={HERON: "heron"})
    taking, withs = _taking(record)
    invited = taking.read("Do you want to watch this video with me?", who=INES, at=20.0)
    assert invited["answer"]["joins"] is False
    assert withs.current(INES) is None


def test_her_answer_travels_to_her_model_as_a_reading_of_the_turn(record):
    from core.brain import what_she_is_given
    from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE

    _lived(record, {INES: 0.5})
    taking, _withs = _taking(record)
    said = "Do you want to watch this video with me?"
    taking.read(said, who=INES, at=20.0)
    block = taking.of_this_turn(said)
    assert block.startswith(HEADER)
    assert "joining is worth more to you than what you would leave for it" in block
    assert taking.of_this_turn("something else entirely") == ""
    what_she_is_given.OF_THIS_TURN.append(taking.of_this_turn)
    try:
        messages, system_prompt = what_she_is_given.for_her_model(
            [{"role": "system", "content": "You are Aura. Be warm."}, {"role": "user", "content": said}],
            "", {}, True, (), said,
        )
    finally:
        what_she_is_given.OF_THIS_TURN.remove(taking.of_this_turn)
    assert system_prompt == "" and [m["role"] for m in messages] == [RUNTIME_EVIDENCE_ROLE, "user"]
    assert HEADER in messages[0]["content"] and "Be warm" not in messages[0]["content"]


def test_a_request_or_a_question_sends_no_such_reading(record):
    taking, _withs = _taking(record)
    for said in ("Watch this video for ten minutes and comment on it.", "Why does a river wind?"):
        taking.read(said, who=INES, at=5.0)
        assert taking.of_this_turn(said) == ""


def test_their_turns_and_hers_are_the_withs_talk_and_leaving_ends_it(record):
    _lived(record, {INES: 0.5})
    taking, withs = _taking(record)
    taking.read("Do you want to watch this video with me?", who=INES, at=20.0)
    held = withs.current(INES)
    taking.she_said("I would. Put it on.", who=INES, at=24.0)
    taking.read("The wrens left the boot this morning.", who=INES, at=60.0)
    taking.she_said("All four at once?", who=INES, at=62.0)
    assert [turn.by for turn in held.turns] == [THEM, HER, THEM, HER]
    assert list(held.silences) == [4.0, 36.0, 2.0]
    # A parting is not read from the words: a greeting does not end a with.
    taking.read("Good morning.", who=INES, at=500.0)
    assert withs.current(INES) is held
    assert taking.left(INES, by=THEM, at=600.0)
    assert withs.current(INES) is None and held.left_by == THEM
    assert not record.thing(INES).under_way
    assert not taking.left(INES, by=THEM, at=601.0)


def test_the_shared_thing_speaking_is_not_a_turn_of_theirs():
    held = With(who=INES, what=FILM, began=0.0)
    ended = _talked(held)
    held.said(IT, ended + 3.0, until=ended + 9.0)
    assert list(held.silences) == [4.0, 6.0, 5.0, 30.0, 8.0]
    assert held.speaking(ended + 5.0) == IT
    assert worth_saying_now(held, ended + 5.0, interest=0.0, spoken_to=True).speak


# ── on her own fast model ───────────────────────────────────────────

HER_FAST_MODEL = Path.home() / ".aura/models/Qwen2.5-1.5B-Instruct-4bit"


@pytest.mark.skipif(not HER_FAST_MODEL.exists(), reason="her fast model is not on this host")
def test_her_own_model_tells_bryans_invitation_from_bryans_task():
    """His two sentences, with nothing told to the model and no system message."""
    from core.brain.llm.what_she_expects_next import ExpectsNext, transcript_of

    assert "system" not in transcript_of("anything at all")
    reader = WhatItAsksOfHer(ExpectsNext(HER_FAST_MODEL).score)
    for said in (
        "Why is the sky blue?", "I got the job.", "Set a timer for twenty minutes.", "Hello Aura.",
        "It snowed here this morning.", "What do limpets eat?", "Close all the windows I have open.",
        "My daughter took her first steps today.", "How long do wrens live?", "Put that album on.",
        "Want to hang out for a bit?", "Bye for now.",
    ):
        assert reader.remember(said)
    invited = reader.heard("Hey do you want to watch this video with me?")
    asked = reader.heard("hey, watch this youtube video for 10 minutes and comment on it as you watch.")
    assert invited.kind == "join" and invited.by_kind["join"] > invited.by_kind["do"]
    assert asked.kind == "do" and asked.by_kind["do"] > asked.by_kind["join"]


# ── through the door a turn actually comes in by ────────────────────


def test_a_turn_with_no_name_on_it_is_from_whoever_she_is_with(record):
    """Live, 13 October: chat came through the kernel, which names nobody, and nothing was read."""
    from core.together.taking_it_up import who_it_is, who_she_is_with

    assert who_she_is_with(record) == who_it_is("", "")
    _lived(record, {INES: 0.5, "person:marek": 0.1}, turns=3)
    assert who_she_is_with(record) == "person:marek"  # the last she met
    record.started(INES, "conversation", label="ines")
    assert who_she_is_with(record) == INES  # whoever she is in the middle of talking with


def test_a_turn_hands_over_the_same_words_twice_and_they_are_read_once(record):
    _lived(record, {INES: 0.5})
    taking, withs = _taking(record)
    said = "Do you want to watch this video with me?"
    first = taking.read(said, who=INES, at=20.0)
    again = taking.read(said, who=INES, at=21.0)
    assert again is first
    assert len(withs.current(INES).turns) == 1


# ── on the desktop the worker that owns the model takes the scores ──


class _TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 50 for char in str(text)]


def _tiny_model(batch):
    """Likes later token ids better, whatever came before them."""
    import mlx.core as mx

    rows, length = batch.shape
    return mx.broadcast_to(mx.arange(50).astype(mx.float32), (rows, length, 50))


class _Steering:
    def __init__(self) -> None:
        self.alpha, self.during = 5.4, []

    def controlled_measurement(self):
        import contextlib

        @contextlib.contextmanager
        def held():
            before = self.alpha
            try:
                yield
            finally:
                self.alpha = before

        return held()

    def set_alpha(self, alpha: float) -> None:
        self.alpha = alpha
        self.during.append(alpha)


def test_the_worker_scores_candidates_unsteered_and_writes_nothing():
    import json
    import threading

    from core.brain.llm.what_she_expects_next import as_a_job_of_the_worker, score_next_turns

    steering = _Steering()
    # "a" is token 47 and "y" is token 21.
    job = {"id": "j1", "action": "generate", "expect_next": ["watch this with me?", "a", "y", "aaa"]}
    answer = as_a_job_of_the_worker(job, _tiny_model, _TinyTokenizer(), steering, threading.Lock())
    scores = json.loads(answer["text"])
    assert answer["status"] == "ok" and answer["id"] == "j1" and answer["tokens_used"] == 0
    assert scores == pytest.approx(score_next_turns(_tiny_model, _TinyTokenizer(), "watch this with me?", ["a", "y", "aaa"]))
    # Later ids are likelier, and a longer candidate is the sum over its tokens,
    # with the padding beside a shorter one counted for nothing.
    assert scores[0] > scores[1]
    assert scores[2] == pytest.approx(3 * scores[0])
    assert steering.during == [0.0] and steering.alpha == 5.4  # stood down for the pass, and back


def test_a_job_with_nothing_to_score_and_a_pass_that_fails_are_both_answered():
    import threading

    from core.brain.llm.what_she_expects_next import as_a_job_of_the_worker

    assert as_a_job_of_the_worker({"id": "j", "expect_next": ["only what was said"]}, _tiny_model, _TinyTokenizer(), None, threading.Lock())["text"] == "[]"

    def broken(batch):
        raise RuntimeError("no room")

    failed = as_a_job_of_the_worker({"id": "j", "expect_next": ["said", "Aa"]}, broken, _TinyTokenizer(), None, threading.Lock())
    assert failed["status"] == "error" and "no room" in failed["message"]


class _TurnsClient:
    """The client a turn holds: answers a scoring job the way the worker does."""

    def __init__(self, kind_of: dict[str, str]) -> None:
        self.asked: list[dict] = []
        self._scorer = _a_model_that_expects(kind_of)

    async def expect_next(self, said, candidates):
        self.asked.append({"said": said, "candidates": list(candidates)})
        return list(self._scorer(said, candidates))


def test_inside_a_turn_the_scores_are_asked_of_the_turns_own_worker(record, monkeypatch):
    import asyncio

    from core.brain import what_she_is_given
    from core.together import taking_it_up

    _lived(record, {INES: 0.5})
    record.started(INES, "conversation", label="ines")
    reader = _reader()
    reader._scorer = lambda said, seconds: (_ for _ in ()).throw(AssertionError("the parent must not score"))
    taking = TakingItUp(reader, record=record, together=Together())
    monkeypatch.setattr(taking_it_up, "_TAKING", taking)
    client = _TurnsClient(SAID)
    said = "Do you want to watch this video with me?"
    try:
        read = asyncio.run(taking_it_up.heard_in_a_turn(said, client))
    finally:
        if taking.of_this_turn in what_she_is_given.OF_THIS_TURN:
            what_she_is_given.OF_THIS_TURN.remove(taking.of_this_turn)
    assert read["proposal"]["kind"] == "join" and read["who"] == INES
    assert client.asked == [{"said": said, "candidates": reader.seconds()}]
    assert taking.of_this_turn(said).startswith(HEADER)


def test_the_real_client_asks_its_worker_with_one_job_that_writes_nothing():
    import asyncio
    import json

    from core.brain.llm.mlx_client import MLXLocalClient

    client = object.__new__(MLXLocalClient)
    sent: list[dict] = []

    async def generate_text_async(prompt, **kwargs):
        sent.append({"prompt": prompt, **kwargs})
        return json.dumps([-3.5, -9.25])

    client.generate_text_async = generate_text_async
    assert asyncio.run(client.expect_next("want to watch this?", ["I'd love to.", "Done."])) == [-3.5, -9.25]
    job = sent[0]
    assert job["expect_next"] == ["want to watch this?", "I'd love to.", "Done."]
    assert job["internal_inference"] is True and job["max_tokens"] == 1 and job["temperature"] == 0.0

    async def gave_the_wrong_count(prompt, **kwargs):
        return json.dumps([-3.5])

    client.generate_text_async = gave_the_wrong_count
    assert asyncio.run(client.expect_next("want to watch this?", ["I'd love to.", "Done."])) == []


def test_a_worker_that_gives_no_scores_leaves_the_turn_as_it_was(record):
    import asyncio

    class Silent:
        async def expect_next(self, said, candidates):
            return []

    taking, withs = _taking(record)
    assert asyncio.run(taking.heard("Do you want to watch this video with me?", who=INES, client=Silent())) == {}
    assert withs.current(INES) is None and taking.of_this_turn("Do you want to watch this video with me?") == ""


def test_what_was_said_is_read_before_her_model_is_handed_the_turn():
    """Live, 13 October: read in the tool step after, her reply said yes to her record's "not now"."""
    gate = Path("core/brain/inference_gate.py").read_text()
    # A turn of her own (a remark while watching) is said by nobody to her, and is not read as said (9e5636d9b).
    reading = gate.index('await self._read_what_was_said_first(_is_user_facing and not context.get("her_own_turn"), '
                         "initial_visible_user_prompt)")
    assert reading < gate.index("messages, system_prompt = for_her_model(", reading)
    serving = Path("core/brain/inference_gate_turn_serving.py").read_text()
    assert 'await heard_in_a_turn(str(said), getattr(self, "_mlx_client", None))' in serving
    worker = Path("core/brain/llm/mlx_worker.py").read_text()
    assert 'if action == "generate" and job.get("expect_next"):' in worker
    assert '"expect_next": [str(c) for c in (kwargs.get("expect_next") or ())]' in Path("core/brain/llm/mlx_client.py").read_text()


def test_the_reading_started_from_a_turn_reaches_the_with_and_her_model(record, monkeypatch):
    import asyncio

    from core.brain import what_she_is_given
    from core.together import taking_it_up

    _lived(record, {INES: 0.5})
    taking, withs = _taking(record)
    monkeypatch.setattr(taking_it_up, "_TAKING", taking)
    said = "Do you want to watch this video with me?"

    async def a_turn():
        taking_it_up.hear_in_the_background(said)
        await asyncio.gather(*list(taking_it_up._HEARING))

    try:
        asyncio.run(a_turn())
        assert taking.of_this_turn in what_she_is_given.OF_THIS_TURN
        assert withs.current(INES) is not None
        assert taking.of_this_turn(said).startswith(HEADER)
    finally:
        if taking.of_this_turn in what_she_is_given.OF_THIS_TURN:
            what_she_is_given.OF_THIS_TURN.remove(taking.of_this_turn)


# ── kept, and used to spare her large model ─────────────────────────


def test_what_the_answers_usually_score_is_learned_once_and_kept(tmp_path, monkeypatch, record):
    from core.runtime.atomic_writer import flush_writes_behind
    from core.together import taking_it_up

    monkeypatch.setattr(taking_it_up, "_file", lambda: tmp_path / "what_it_asks_of_her.json")
    assert taking_it_up._stored() == {}
    taking = TakingItUp(_reader(heard_before=12), record=record, together=Together())
    taking._keeps = True
    taking._seeded = True
    taking.read("Do you want to watch this video with me?", who=INES, at=1.0)
    flush_writes_behind()
    kept = taking_it_up._stored()
    assert kept["heard"] == 13 and len(kept["usual"]) == sum(len(SECONDS[kind]) for kind in KINDS)
    again = WhatItAsksOfHer(_a_model_that_expects(SAID), kept)
    assert again.heard("Watch this video for ten minutes and comment on it.").kind == "do"


def test_a_telling_or_a_greeting_is_not_looked_up_in_what_she_can_do(record, monkeypatch):
    import asyncio

    from core.together import taking_it_up

    taking, _withs = _taking(record)
    monkeypatch.setattr(taking_it_up, "_TAKING", taking)
    for said, asked in (
        ("The wrens left the boot this morning.", False),
        ("Do you want to watch this video with me?", True),
        ("Good morning.", False),
        ("Watch this video for ten minutes and comment on it.", True),
        ("Why does a river wind?", True),
    ):
        taking.read(said, who=INES, at=1.0)
        assert asyncio.run(taking_it_up.asked_to_do_something(said)) is asked, said
    # What was never read may be a request.
    assert asyncio.run(taking_it_up.asked_to_do_something("something nobody read")) is True


def test_a_reading_still_under_way_is_waited_for(record, monkeypatch):
    import asyncio

    from core.together import taking_it_up

    taking, _withs = _taking(record)
    monkeypatch.setattr(taking_it_up, "_TAKING", taking)
    said = "The wrens left the boot this morning."

    async def a_turn():
        taking_it_up.hear_in_the_background(said, person="ines", sent_from="desk")
        return await taking_it_up.asked_to_do_something(said)

    try:
        assert asyncio.run(a_turn()) is False
    finally:
        from core.brain import what_she_is_given

        if taking.of_this_turn in what_she_is_given.OF_THIS_TURN:
            what_she_is_given.OF_THIS_TURN.remove(taking.of_this_turn)


def test_the_scorer_does_not_load_a_model_where_mlx_is_kept_off_the_gpu(monkeypatch):
    """Live, 13 October: it loaded in the desktop's parent and ground one core on the CPU."""
    import mlx.core as mx

    from core.brain.llm.what_she_expects_next import ExpectsNext

    if not HER_FAST_MODEL.exists():
        pytest.skip("her fast model is not on this host")
    monkeypatch.setattr(mx, "default_device", lambda: mx.cpu)
    ear = ExpectsNext(HER_FAST_MODEL)
    assert ear.score("Do you want to watch this with me?", ["I'd love to."]) == []
    assert ear.status() == {"loaded": False, "refused": "this process keeps MLX off the GPU", "model": HER_FAST_MODEL.name}
    # And with nothing scored, nothing is read and nothing is waited on twice.
    reader = WhatItAsksOfHer(ear.score)
    assert reader.heard("Do you want to watch this with me?").kind == ""


def test_a_reader_with_no_history_is_set_against_what_follows_nothing_in_particular(record, monkeypatch):
    """Her cortex starts with nothing to set its scores against, and four passes give it a reference."""
    import asyncio

    from core.together import taking_it_up
    from core.together.what_was_proposed import CONTENT_FREE

    _lived(record, {INES: 0.5})
    record.started(INES, "conversation", label="ines")
    fresh = WhatItAsksOfHer(lambda said, seconds: (_ for _ in ()).throw(AssertionError("the parent must not score")))
    taking = TakingItUp(fresh, record=record, together=Together())
    monkeypatch.setattr(taking_it_up, "_TAKING", taking)
    client = _TurnsClient(SAID)
    said = "Do you want to watch this video with me?"
    try:
        read = asyncio.run(taking_it_up.heard_in_a_turn(said, client))
    finally:
        from core.brain import what_she_is_given

        if taking.of_this_turn in what_she_is_given.OF_THIS_TURN:
            what_she_is_given.OF_THIS_TURN.remove(taking.of_this_turn)
    assert [asked["said"] for asked in client.asked] == [*CONTENT_FREE, said]
    assert read["proposal"]["kind"] == "join" and read["proposal"]["heard_before"] == len(CONTENT_FREE)


# ── what she can sense and reach, and what is on the screen ─────────


class _Seen:
    def __init__(self, summary, known=True):
        self.summary, self.known = summary, known

    def as_evidence_line(self):
        return self.summary


def _senses_and_screen(monkeypatch, taking_it_up, screen="Safari — Herons at dawn — YouTube\n0:42 / 3:10"):
    from core.self import capability_ledger

    class Ledger:
        def measure_all(self):
            return {
                "screen_sight": _Seen("I can capture and read this screen."),
                "world_access": _Seen("I can reach the web."),
                "camera": _Seen("unknown", known=False),
            }

    monkeypatch.setattr(capability_ledger, "get_capability_ledger", lambda: Ledger())

    async def the_screen_now():
        return screen

    monkeypatch.setattr(taking_it_up, "_the_screen_now", the_screen_now)


def test_an_invitation_carries_what_she_can_sense_and_what_is_on_the_screen(record, monkeypatch):
    """Live, 13 October: invited to watch a video in his browser, she denied having screen or web."""
    import asyncio

    from core.together import taking_it_up

    _lived(record, {INES: 0.5})
    record.started(INES, "conversation", label="ines")
    taking = TakingItUp(_reader(), record=record, together=Together())
    monkeypatch.setattr(taking_it_up, "_TAKING", taking)
    _senses_and_screen(monkeypatch, taking_it_up)
    said = "Do you want to watch this video with me?"
    try:
        asyncio.run(taking_it_up.heard_in_a_turn(said, _TurnsClient(SAID)))
    finally:
        from core.brain import what_she_is_given

        if taking.of_this_turn in what_she_is_given.OF_THIS_TURN:
            what_she_is_given.OF_THIS_TURN.remove(taking.of_this_turn)
    given = taking.of_this_turn(said)
    assert given.startswith(HEADER)
    assert f"{taking_it_up.SENSES_HEADER}\n- I can capture and read this screen.\n- I can reach the web." in given
    assert "unknown" not in given  # what could not be measured is not said
    assert f"{taking_it_up.SCREEN_HEADER}\nSafari — Herons at dawn — YouTube" in given
    # And the with she joined has the screen as its shared field.
    held = taking._withs().current(INES)
    assert held.in_view(time.time()).what == "Safari — Herons at dawn — YouTube"


def test_a_request_carries_her_senses_and_a_telling_carries_nothing(record, monkeypatch):
    import asyncio

    from core.together import taking_it_up

    taking, _withs = _taking(record)
    monkeypatch.setattr(taking_it_up, "_TAKING", taking)
    _senses_and_screen(monkeypatch, taking_it_up)
    asked = "Watch this video for ten minutes and comment on it."
    taking.read(asked, who=INES, at=1.0)
    asyncio.run(taking.take_what_it_needs())
    given = taking.of_this_turn(asked)
    assert taking_it_up.SENSES_HEADER in given and taking_it_up.SCREEN_HEADER not in given
    told = "The wrens left the boot this morning."
    taking.read(told, who=INES, at=2.0)
    asyncio.run(taking.take_what_it_needs())
    assert taking.of_this_turn(told) == ""


def test_what_the_screen_shows_moves_the_withs_field_by_how_much_it_changed(record, monkeypatch):
    from core.together.taking_it_up import _how_much_changed

    held = With(who=INES, what=FILM, began=0.0)
    assert _how_much_changed(held, "a heron stands in the reeds") is None
    assert _how_much_changed(held, "a heron stands in the reeds") == 0.0
    assert _how_much_changed(held, "credits roll now") == 1.0
    assert _how_much_changed(held, "credits roll later") == pytest.approx(0.5)


def test_what_she_is_unsure_of_is_counted_for_each_side():
    """The same rule her wants are ranked by: liking plus the root of its variance."""
    import types

    from core.together import her_answer as answering

    class Held:
        def __init__(self, key, liking, unsure, contacts=3, under_way=None):
            self.key, self.liking, self.unsure, self.contacts = key, liking, unsure, contacts
            self.label, self.under_way, self.open = key, under_way or {}, {}
            self.during = types.SimpleNamespace(times=0, liking=0.0)
            self.kind = key.split(":", 1)[0]

    def record_with(person_unsure):
        things = {
            INES: Held(INES, 0.0, person_unsure),
            HERON: Held(HERON, 0.1, 0.0001, under_way={"reading": 1}),
        }
        return types.SimpleNamespace(
            _things=things, thing=things.get, decided=lambda: None, known=lambda key: None,
        )

    assert answering.her_answer(record_with(0.0001), INES, []).joins is False  # well known: the mean decides
    assert answering.her_answer(record_with(0.25), INES, []).joins is True  # little known: the doubt counts for her


def test_a_thing_she_knows_she_dislikes_is_still_not_now_whatever_she_is_unsure_of(record):
    for turn in range(40):
        key, worth = ((FILM, -0.6), (HERON, 0.6))[turn % 2]
        record.met([key], worth, reachable=[key], labels={key: key.split(":", 1)[1]})
    record.met([INES], 0.1, reachable=[INES], labels={INES: "ines"})
    assert her_answer(record, INES, [FILM]).joins is False


# ── remarks while watching together ─────────────────────────────────


class _Screen:
    def __init__(self, capture, source="Safari"):
        self.capture, self.source = capture, source


def _watching_with_ines(monkeypatch, *, said_since=True, interest_of=0.9):
    from core.conversation import persistence
    from core.together import remarks

    held = With(who=INES, what=FILM, began=1000.0)
    _talked(held, start=1000.0)
    held.remarks, held.taken_up = 6, 6
    stored = [
        {"role": "user", "content": "Do you want to watch this with me?", "created_at": 1000.5},
        {"role": "aura", "content": "Count me in.", "created_at": 1001.0},
    ] if said_since else []

    class Store:
        def get_recent_history(self, limit=256):
            return stored

    monkeypatch.setattr(persistence, "get_persistence", lambda: Store())
    monkeypatch.setattr(remarks, "_interest", lambda capture, change: (interest_of, HERON))
    asked: list[dict] = []

    class Router:
        async def think(self, prompt, **kwargs):
            asked.append({"prompt": prompt, **kwargs})
            return (True, "Look at how still it stands before the strike.", {})

    from core.container import ServiceContainer

    monkeypatch.setattr(
        ServiceContainer, "get",
        classmethod(lambda cls, name, default=None: Router() if name == "llm_router" else default),
    )
    return held, asked


def test_in_a_lull_she_says_something_about_what_they_are_watching(monkeypatch):
    import asyncio

    from core.together.remarks import remark_in_a_with
    from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE

    held, asked = _watching_with_ines(monkeypatch)
    later = held.turns[-1].until + 60.0
    said = asyncio.run(remark_in_a_with(held, _Screen("A heron stands in the shallows\n1:12 / 3:10"), now=later))
    assert said == "Look at how still it stands before the strike."
    job = asked[0]
    assert job["system_prompt"] == "" and job["is_background"] is True
    assert [m["role"] for m in job["messages"]] == ["user", "assistant", RUNTIME_EVIDENCE_ROLE]
    assert job["messages"][-1]["content"].startswith("## WHAT IS ON THE SCREEN\nSafari\nA heron stands")
    # The field moved, and her remark is a turn of the with that waits to be taken up.
    assert held.in_view(later).what == "A heron stands in the shallows"
    assert held.turns[-1].by == HER and held.remarks == 7


def test_she_does_not_talk_over_him_and_the_field_still_moves(monkeypatch):
    import asyncio

    from core.together.remarks import remark_in_a_with

    held, asked = _watching_with_ines(monkeypatch)
    now = held.turns[-1].until + 30.0
    held.said(THEM, now - 1.0)
    assert asyncio.run(remark_in_a_with(held, _Screen("The heron strikes"), now=now)) == ""
    assert asked == [] and held.in_view(now).what == "The heron strikes"


def test_nothing_of_interest_is_not_said(monkeypatch):
    import asyncio

    from core.together.remarks import remark_in_a_with

    held, asked = _watching_with_ines(monkeypatch, interest_of=0.0)
    assert asyncio.run(remark_in_a_with(held, _Screen("Loading..."), now=held.turns[-1].until + 60.0)) == ""
    assert asked == []


def test_with_no_turn_of_theirs_in_the_with_there_is_no_turn_of_hers_to_write(monkeypatch):
    import asyncio

    from core.together.remarks import remark_in_a_with

    held, asked = _watching_with_ines(monkeypatch, said_since=False)
    assert asyncio.run(remark_in_a_with(held, _Screen("A heron"), now=held.turns[-1].until + 60.0)) == ""
    assert asked == []


def test_while_a_with_is_open_her_screen_looks_go_to_it():
    source = Path("core/perception/ambient_presence.py").read_text()
    body = source[source.index("async def _consider_speaking"):]
    body = body[:body.index("if thought and self.offer_utterance(thought):")]
    assert "held = get_together().current()" in body
    assert "remark_in_a_with(held, latest_screen()) if held is not None else consider_utterance(result)" in body
    assert "held is None and time.time() - self._last_spoke_at < self._min_speech_gap_s" in body
