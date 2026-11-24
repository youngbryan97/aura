"""What a turn asks for is read from states, decided by cost, held for every door, and carried out until a check says so."""
from __future__ import annotations

import asyncio
import json
import re
import time

import numpy as np
import pytest

from core.brain.llm import her_states_at_a_turn as states
from core.runtime import what_this_turn_asks as held
from core.runtime.what_this_turn_asks import Asked, asked_by, hold, licensed
from core.understanding import carrying_out, what_is_asked
from core.understanding.effects import BY_NAME, EFFECTS
from core.understanding.examples import Turn, attached, examples, transcript
from core.understanding.requests_of_her import (
    CUT_SHORT,
    MET,
    PARTLY,
    UNMET,
    WORKING,
    Part,
    Request,
    RequestsOfHer,
)
from core.understanding.taking_up import settle
from core.understanding.what_is_asked import WhatIsAsked, decide, fit


@pytest.fixture(autouse=True)
def _nothing_held():
    held.clear()
    held._BY_WORDS.clear()
    yield
    held.clear()
    held._BY_WORDS.clear()


def _read(said: str, effect: str, decision: str = "act", **p: float) -> Asked:
    return Asked(said=said, effect=effect, decision=decision, sure=0.9, asking=0.9, by_effect=p or {effect: 0.9, "talk": 0.1},
                 at=time.time())


# ── the examples ─────────────────────────────────────────────────────

def test_every_example_is_of_a_known_kind_and_none_contradicts_another():
    seen: dict[tuple, str] = {}
    for example in examples():
        assert example.effect in BY_NAME
        key = (example.said, example.before)
        assert seen.setdefault(key, example.effect) == example.effect, key
    by_kind = {effect.name: sum(1 for e in examples() if e.effect == effect.name) for effect in EFFECTS}
    assert min(by_kind.values()) >= 15, by_kind


def test_no_example_holds_anything_that_could_be_a_persons():
    text = "\n".join([e.said for e in examples()] + [t.said for e in examples() for t in e.before])
    assert not re.search(r"[\w.]+@[\w.]+\.\w+", text)
    assert not re.search(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", text)


def test_the_same_words_are_both_kinds_by_what_came_before():
    kinds = {}
    for example in examples():
        if example.before:
            kinds.setdefault(example.said, set()).add(example.effect)
    both = [said for said, k in kinds.items() if "talk" in k and len(k) > 1]
    assert "Could you build a chess clock?" in both
    assert len(both) >= 5


def test_a_transcript_is_the_exchange_and_nothing_else():
    text = transcript("Could you?", [Turn("user", "I drew a map."), Turn("assistant", "Show me!")])
    assert text == ("<|im_start|>user\nI drew a map.<|im_end|>\n<|im_start|>assistant\nShow me!<|im_end|>\n"
                    "<|im_start|>user\nCould you?<|im_end|>\n<|im_start|>assistant\n")
    assert "system" not in text
    long = transcript("start " + "word " * 2000 + "end")
    assert "start" in long and "end<|im_end|>" in long and re.search(r"\[pasted text, \d+ words\]", long)
    assert attached("Find me work.", ["cv.pdf"]) == "Find me work. [attached: cv.pdf]"


# ── the states ───────────────────────────────────────────────────────

class _Layer:
    is_linear = True

    def __call__(self, x):
        return x + 1.0


class _Model:
    def __init__(self, layers):
        self.layers = layers
        self.model = self

    def __call__(self, ids):
        import mlx.core as mx

        x = mx.broadcast_to(ids.astype(mx.float32)[..., None], (*ids.shape, 64))
        for layer in self.layers:
            assert layer.is_linear
            x = layer(x)
        return x


class _Tokens:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 97 for c in text]


def test_states_are_read_at_the_last_position_and_the_layers_are_put_back(monkeypatch):
    pytest.importorskip("mlx.core")
    layers = [_Layer() for _ in range(8)]
    model = _Model(layers)
    monkeypatch.setattr(states, "_layers", lambda m: m.layers)
    text = json.dumps(states.states_where_she_answers(model, _Tokens(), ["abc", "abd"]))
    assert model.layers is layers and all(isinstance(layer, _Layer) for layer in layers)
    read = states.states_from_text(text, 2)
    assert len(read) == 2 and read[0].shape == (len(states.DEPTHS), states.SKETCH)
    again = states.states_from_text(json.dumps(states.states_where_she_answers(model, _Tokens(), ["abc"])), 1)
    assert np.allclose(read[0], again[0])
    assert not np.allclose(read[0], read[1])


def test_the_layers_are_put_back_when_the_pass_breaks(monkeypatch):
    pytest.importorskip("mlx.core")

    class Breaks(_Model):
        def __call__(self, ids):
            raise RuntimeError("no room")

    layers = [_Layer() for _ in range(8)]
    model = Breaks(layers)
    monkeypatch.setattr(states, "_layers", lambda m: m.layers)
    response = states.as_a_job_of_the_worker({"id": "x", "states_at_turn": ["abc"]}, model, _Tokens(), None, _Nothing())
    assert response["status"] == "error" and "no room" in response["message"]
    assert all(isinstance(layer, _Layer) for layer in layers)


class _Nothing:
    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def test_a_sketch_keeps_which_states_are_near():
    rng = np.random.default_rng(3)
    a = rng.normal(size=5000)
    near, far = a + 0.1 * rng.normal(size=5000), rng.normal(size=5000)

    def sketched(v):
        return states.states_from_text(json.dumps([[states._sketched(v)] * len(states.DEPTHS)]), 1)[0][0]

    sa, sn, sf = sketched(a), sketched(near), sketched(far)
    cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))  # noqa: E731
    assert cos(sa, sn) > 0.95 and abs(cos(sa, sf)) < 0.2


@pytest.mark.parametrize("text", ["", "not json", "[]", '[["AA=="]]', '{"a":1}'])
def test_what_is_not_states_reads_as_nothing(text):
    assert states.states_from_text(text, 1) == []


# ── the fit, and what is done about it ───────────────────────────────

def _blobs(kinds: list[str], per: int, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    centres = {kind: rng.normal(size=(2, 40)) * 3 for kind in kinds}
    X, y = [], []
    for kind in kinds:
        for _ in range(per):
            X.append(centres[kind] + rng.normal(size=(2, 40)))
            y.append(kind)
    return np.stack(X).astype(np.float32), y


def test_the_fit_says_how_it_did_on_what_it_was_not_fitted_on():
    X, y = _blobs(["talk", "program", "found"], 20)
    fitted = fit(X, y)
    assert fitted.over == 60 and fitted.told > 0.9 and fitted.told_asking > 0.9
    p = fitted.probabilities(X[0])
    assert abs(sum(p.values()) - 1.0) < 1e-6 and max(p, key=p.get) == "talk"


def test_a_reading_that_is_not_ready_answers_nothing(tmp_path):
    class Client:
        model_path = str(tmp_path)

        async def states_at_turn(self, transcripts, *, in_a_turn):
            raise AssertionError("not asked")

    assert asyncio.run(WhatIsAsked(root=tmp_path).read("Make me a clock.", client=Client())) is None


def test_it_is_fitted_once_for_a_model_and_kept(tmp_path):
    kinds = ["talk", "program"]
    given = [what_is_asked.Example(kind, f"{kind} {n}") for kind in kinds for n in range(12)]
    asked: list[int] = []

    class Client:
        model_path = str(tmp_path / "model")

        async def states_at_turn(self, transcripts, *, in_a_turn):
            asked.append(len(transcripts))
            out = []
            for text in transcripts:
                v = np.zeros((len(states.DEPTHS), 16), np.float32)
                v[:, 0 if "talk" in text else 1] = 4.0
                v[:, 2:] = np.random.default_rng(len(text)).normal(size=(len(states.DEPTHS), 14))
                out.append(v)
            return out

    reads = WhatIsAsked(root=tmp_path, given=given)
    assert asyncio.run(reads.learn(Client())) and reads.ready(Client())
    assert sum(asked) == 24
    second = WhatIsAsked(root=tmp_path, given=given)
    assert asyncio.run(second.learn(Client()))
    assert sum(asked) == 24, "what was read once is not read again"
    got = asyncio.run(second.read("program 99", client=Client()))
    assert got is not None and got.effect == "program" and got.acts


def test_a_batch_waits_for_room_and_for_whoever_is_talking(monkeypatch):
    slept: list[float] = []

    async def sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(what_is_asked.asyncio, "sleep", sleep)
    answers = iter([[], ["state"]])
    talking = iter([True, False, False])

    async def ask(texts, *, in_a_turn):
        assert in_a_turn
        return next(answers)

    got = asyncio.run(WhatIsAsked._read_when_it_can(["x"], ask, lambda: next(talking)))
    assert got == ["state"] and slept == [1.0, 2.0]

    async def never(texts, *, in_a_turn):
        return []

    assert asyncio.run(WhatIsAsked._read_when_it_can(["x"], never, None)) == []


@pytest.mark.parametrize(("p", "decision", "effect"), [
    ({"talk": 0.97, "program": 0.03}, "talk", "talk"),
    ({"talk": 0.02, "program": 0.98}, "act", "program"),
    # As likely a question about her as a request: one question settles it.
    ({"talk": 0.40, "program": 0.60}, "ask_first", "program"),
    # Reading what they gave her costs nothing, so it is done where building would be asked about.
    ({"talk": 0.40, "document_read": 0.60}, "act", "document_read"),
    # What reaches somebody else wants more certainty than what does not.
    ({"talk": 0.15, "sent": 0.85}, "ask_first", "sent"),
    ({"talk": 0.15, "found": 0.85}, "act", "found"),
    # A little of everything is nothing in particular.
    ({"talk": 0.64, "found": 0.08, "known": 0.08, "program": 0.07, "written": 0.07, "status": 0.06}, "talk", "talk"),
])
def test_what_is_done_is_what_costs_least(p, decision, effect):
    kind, what, asking = decide(p)
    assert (what, kind) == (decision, effect)
    assert asking == pytest.approx(1 - p["talk"])


# ── every door sees the reading ──────────────────────────────────────

def test_a_reading_is_found_by_its_words_and_by_parts_of_them():
    asked = _read("Make me a tide clock. Put it in Applications.", "program")
    hold(asked)
    assert asked_by("Make me a tide clock.") is asked
    assert asked_by("[context]\nUser message: Make me a tide clock. Put it in Applications.") is asked
    assert asked_by("Something else entirely") is None
    assert licensed(asked.said, "program") is True and licensed(asked.said, "found") is False
    held.clear()
    assert asked_by(asked.said) is asked, "by its words, where the turn's context is gone"
    assert licensed("never read") is None


def test_a_word_is_not_found_by_accident():
    hold(_read("yes", "resume"))
    assert asked_by("no, yesterday I said yes to them both") is None
    assert asked_by("User message: yes") is not None


def test_the_desktop_lane_takes_what_was_read_and_not_what_was_worded():
    from core.runtime.desktop_objective_intent import (
        asks_to_build_software,
        looks_like_desktop_objective,
    )

    said = "Could you build me Lanternfall for my desktop? Just for fun."
    hold(_read(said, "program"))
    assert asks_to_build_software(said) and not looks_like_desktop_objective(said)
    said = "Pop the calculator up."
    hold(_read(said, "on_machine"))
    assert looks_like_desktop_objective(said) and not asks_to_build_software(said)
    said = "I opened an app company with my brother and we built a game."
    hold(_read(said, "talk", "talk"))
    assert not looks_like_desktop_objective(said) and not asks_to_build_software(said)


def test_no_page_is_worked_for_a_turn_that_asks_for_no_page():
    from core.conversation.page_interaction import page_interaction_target

    said = "Find me openings I could get and link me to them. Sign in is not needed. me at example.org"
    hold(_read(said, "found"))
    assert page_interaction_target(said) == ""


def test_the_reconstruction_lane_is_not_taken_by_a_quoted_title():
    from interface.routes.chat_capability_inventory import _looks_like_program_dna_execution_request

    said = 'Find me work. I wrote "The Rebuilt Clock" and reconstructed a research software tool for a clean-room lab.'
    hold(_read(said, "found"))
    assert not _looks_like_program_dna_execution_request(said)


# ── what she was asked, and how it stands ────────────────────────────

def test_requests_are_kept_and_work_under_way_is_read_back_as_cut_short(tmp_path):
    ledger = RequestsOfHer(tmp_path / "requests.json")
    one = ledger.asked("Make a tide clock.", effect="program", conversation="c")
    asyncio.run(ledger.changed(one, stands=WORKING, now="writing it"))
    again = RequestsOfHer(tmp_path / "requests.json")
    read = again.get(one.id)
    assert read is not None and read.stands == CUT_SHORT
    assert again.unfinished("c") is read and again.unfinished("other") is None


def test_an_account_names_every_part_that_is_missing_and_why():
    request = Request(id="a", conversation="c", said="Make it.", effect="program", stands=PARTLY, opened_at=time.time(),
                      parts=[Part("it moves", "met", "3 of 3 checks hold"), Part("it scores", "unmet", "no version passed")])
    account = "\n".join(request.account())
    assert "partly done" in account and "it moves: done (3 of 3 checks hold)" in account
    assert "it scores: not done (no version passed)" in account


def _p(**p: float) -> dict[str, float]:
    return p


def test_status_with_nothing_asked_for_is_the_next_likeliest_kind(tmp_path):
    ledger = RequestsOfHer(tmp_path / "r.json")
    asked = Asked(said="How are the games going?", effect="status", decision="act", by_effect=_p(status=0.5, talk=0.48, found=0.02))
    settled = settle(asked, ledger, conversation="c")
    assert settled.asked.decision == "talk" and settled.request is None


def test_where_is_it_and_try_again_are_about_what_was_asked(tmp_path):
    ledger = RequestsOfHer(tmp_path / "r.json")
    request = ledger.asked("Make a tide clock.", effect="program", conversation="c")
    request.stands = UNMET
    asked = Asked(said="where is it?", effect="status", decision="act", by_effect=_p(status=0.97, talk=0.03))
    assert settle(asked, ledger, conversation="c").request is request
    asked = Asked(said="try again", effect="resume", decision="act", by_effect=_p(resume=0.96, talk=0.04))
    settled = settle(asked, ledger, conversation="c")
    assert settled.request is request and settled.what == "on_with" and settled.asked.effect == "program"
    request.stands = WORKING
    assert settle(asked, ledger, conversation="c").what == "told_more"


def test_going_on_with_what_was_asked_is_not_asked_about(tmp_path):
    ledger = RequestsOfHer(tmp_path / "r.json")
    asked = Asked(said="Continue on", effect="resume", decision="ask_first", by_effect=_p(resume=0.45, talk=0.49, program=0.06))
    assert settle(asked, ledger, conversation="c").asked.decision == "talk"
    request = ledger.asked("Make a tide clock.", effect="program", conversation="c")
    request.stands = CUT_SHORT
    settled = settle(asked, ledger, conversation="c")
    assert settled.request is request and settled.what == "on_with"


def test_more_of_the_same_request_is_the_same_request(tmp_path):
    ledger = RequestsOfHer(tmp_path / "r.json")
    request = ledger.asked("Find me openings.", effect="found", conversation="c")
    request.stands = WORKING
    asked = Asked(said="Remote, and near the coast.", effect="found", decision="act", by_effect=_p(found=0.97, talk=0.03))
    settled = settle(asked, ledger, conversation="c", before=["Find me openings."])
    assert settled.request is request and settled.what == "told_more"
    other = settle(asked, ledger, conversation="c", before=["I like the sea."])
    assert other.request is None and other.what == "new"


# ── carried out until a check says so ────────────────────────────────

def _carried(tmp_path, carrier, said="Make it."):
    ledger = RequestsOfHer(tmp_path / "r.json")
    request = ledger.asked(said, effect="program", conversation="c")
    told: list[str] = []
    asyncio.run(carrying_out.carry_out(request, carrier, tell=told.append, ledger=ledger))
    return request, told


def test_a_carrier_saying_so_does_not_make_it_so(tmp_path):
    async def carrier(request, tell):
        return carrying_out.CameTo(how="said it was done", came_to="Done!")

    request, _told = _carried(tmp_path, carrier)
    assert request.stands == UNMET and request.missing()


def test_it_tries_again_while_that_gains_and_names_what_is_left(tmp_path):
    left = iter([["b", "c"], ["c"], ["c"], []])

    async def carrier(request, tell):
        missing = next(left)
        return carrying_out.CameTo(how="tried", parts=[Part(w, "unmet" if w in missing else "met", "it fails its check")
                                                       for w in ("a", "b", "c")])

    request, told = _carried(tmp_path, carrier)
    assert len(request.tries) == 3 and request.stands == PARTLY
    assert [p.what for p in request.missing()] == ["c"]
    assert any("b; c" in line for line in told)


def test_a_try_that_breaks_is_named_and_not_silent(tmp_path):
    async def carrier(request, tell):
        raise OSError("the disk is full")

    request, _told = _carried(tmp_path, carrier)
    assert request.stands == UNMET
    assert "the disk is full" in "\n".join(request.account())


def test_met_only_when_every_part_is(tmp_path):
    async def carrier(request, tell):
        return carrying_out.CameTo(how="built", parts=[Part("it runs", "met"), Part("it is where it was asked", "met")],
                                   made={"app": "/Applications/Tide Clock.app"})

    request, _told = _carried(tmp_path, carrier)
    assert request.stands == MET and "/Applications/Tide Clock.app" in "\n".join(request.account())


def test_stopped_work_is_cut_short(tmp_path):
    async def run():
        ledger = RequestsOfHer(tmp_path / "r.json")
        request = ledger.asked("Make it.", effect="program", conversation="c")

        async def carrier(request, tell):
            await asyncio.sleep(60)

        task = asyncio.create_task(carrying_out.carry_out(request, carrier, tell=lambda line: None, ledger=ledger))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return request

    assert asyncio.run(run()).stands == CUT_SHORT


# ── what it came to, told ────────────────────────────────────────────

def test_her_words_stand_only_where_they_name_what_there_is():
    from interface.routes.chat_carrying_out import the_message

    request = Request(id="a", conversation="c", said="Make it.", effect="program", stands=MET, opened_at=time.time(),
                      parts=[Part("it runs", "met")], made={"app": "/Applications/TideClock.app"})
    assert the_message(request, "It's at /Applications/TideClock.app, have a go.") == "It's at /Applications/TideClock.app, have a go."
    assert "It stands: done" in the_message(request, "It's at /Applications/Other.app.")
    assert "It stands: done" in the_message(request, "")
    request.made["told"] = "1. [A](https://a.example/1)"
    assert the_message(request, "Found one.").endswith("1. [A](https://a.example/1)")


def test_taking_up_opens_begins_and_records_the_call(tmp_path, monkeypatch):
    from interface.routes import chat_carrying_out as chat

    ledger = RequestsOfHer(tmp_path / "r.json")
    monkeypatch.setattr(chat, "get_requests_of_her", lambda: ledger)
    recorded: list[dict] = []
    monkeypatch.setattr(chat, "_recorded", lambda said, returned, ok=True: recorded.append(returned))

    async def before(session, said):
        return []

    monkeypatch.setattr(chat, "_the_words_before", before)
    begun: list[Request] = []
    monkeypatch.setattr(carrying_out, "begin", lambda request, **kw: begun.append(request) or True)
    said = "Could you build me Lanternfall?"

    async def turn():
        hold(_read(said, "program"))
        return await chat.take_up_what_is_asked(said, session_id="c"), held.this_turn().request_id

    taken, request_id = asyncio.run(turn())
    assert taken and begun and begun[0].said == said and recorded[-1]["begun"] is True
    assert request_id == begun[0].id

    held.clear()
    hold(_read("It's a lovely evening.", "talk", "talk"))
    assert not asyncio.run(chat.take_up_what_is_asked("It's a lovely evening.", session_id="c"))
    assert len(begun) == 1

    hold(_read("where is it?", "status"))
    assert asyncio.run(chat.take_up_what_is_asked("where is it?", session_id="c"))
    assert "Lanternfall" in json.dumps(recorded[-1])


def test_the_turn_is_read_before_any_door_and_taken_up_before_any_lane():
    import inspect

    from interface.routes import chat

    body = inspect.getsource(chat._api_chat_turn)
    assert body.index("read_what_this_turn_asks(") < body.index("_api_chat_turn__grounded_recall_context(")
    assert body.index("take_up_what_is_asked(") < body.index("_resolve_action_episode_projection(")
    assert body.index("take_up_what_is_asked(") < body.index("_execute_governed_capability_request_from_chat(")
    assert "and not taken_up" in body
