"""Her voice mode keeps one conversation her model reads once and only adds to (core/voice/duplex/live_context.py).

LIVE 11 November 2026: "What's two plus two?" read 2,354 tokens to say "Four." On her cortex the live context gave a
first token 0.32 s after the words, against 5.4 to 6.7 s rebuilt (tools/voice/prove_a_live_context.py).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.utils.injected_blocks import stamp_runtime_payload
from core.voice.duplex import live_context as lc


def _said(user, aura, at=None):
    """An exchange as her runtime records it: stamped as its own, with when it was completed."""
    import time

    return stamp_runtime_payload({"user": user, "aura": aura, "timestamp": time.time() if at is None else at})


def _a_context(replies=("Four.",)):
    read, generated, recorded = [], [], []

    async def read_fn(text, scope):
        read.append((text, scope))
        return {"status": "ok", "held": len(text)}

    answers = list(replies)

    async def generate_fn(prompt, *, scope, words, reply_stream, timeout_s):
        generated.append((prompt, scope, words))
        return answers.pop(0) if answers else None

    async def record_fn(words, reply):
        recorded.append((words, reply))

    live = lc.HerLiveContext("session-1", read=read_fn, generate=generate_fn, record=record_fn)
    return live, read, generated, recorded


def test_the_conversation_is_read_once_and_each_turn_only_adds_to_it(monkeypatch):
    monkeypatch.setattr(lc, "the_turns_evidence", lambda: "[EVIDENCE]\n")

    async def scenario():
        live, read, generated, recorded = _a_context(replies=("Four.", "Six."))
        await live.open([_said("Hi.", "Hello.")])
        head = live.text
        assert head == "<|im_start|>user\nHi.<|im_end|>\n<|im_start|>assistant\nHello.<|im_end|>\n"
        assert read == [(head, "voice_live:session-1")] and live.ready
        live.a_person_began_to_speak()
        await asyncio.sleep(0)
        await live.answer("What's two plus two?")
        first_prompt = generated[0][0]
        assert first_prompt.startswith(head + "<|im_start|>user\n[EVIDENCE]\n"), "the turn begins where the context ends"
        assert read[1][0] == head + "<|im_start|>user\n[EVIDENCE]\n", "the evidence was read while they spoke"
        assert first_prompt.endswith("What's two plus two?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert live.text == first_prompt + "Four.<|im_end|>\n", "her reply goes in as she said it"
        live.a_person_began_to_speak()
        await asyncio.sleep(0)
        await live.answer("And three plus three?")
        assert generated[1][0].startswith(live.text[: len(first_prompt + "Four.<|im_end|>\n")])
        assert recorded == [("What's two plus two?", "Four."), ("And three plus three?", "Six.")]

    asyncio.run(scenario())


def test_until_the_conversation_is_read_every_turn_takes_her_governed_route():
    live, _read, _generated, _recorded = _a_context()
    assert live.answers("What's two plus two?") is False


def test_a_turn_that_asks_her_to_act_or_about_her_runtime_takes_her_governed_route(monkeypatch):
    live, *_ = _a_context()
    live.ready = True
    monkeypatch.setattr(lc, "goes_to_her_governed_route", lambda text: "open" in text)
    assert live.answers("Can you open Safari for me?") is False
    assert live.answers("What's two plus two?") is True


def test_a_turn_her_governed_route_answered_is_added_to_the_context():
    live, *_ = _a_context()
    live.ready = True
    live.heard_elsewhere("Can you open Safari?", "Opened it.")
    assert live.text.endswith("<|im_start|>user\nCan you open Safari?<|im_end|>\n<|im_start|>assistant\nOpened it.<|im_end|>\n")


def test_an_answer_that_did_not_form_leaves_the_context_as_it_was(monkeypatch):
    monkeypatch.setattr(lc, "the_turns_evidence", lambda: "")

    async def scenario():
        live, _read, _generated, recorded = _a_context(replies=())
        await live.open([])
        before = live.text
        assert await live.answer("Hello?") is None
        assert live.text == before and recorded == []

    asyncio.run(scenario())


def test_a_context_that_could_not_be_read_stays_closed():
    async def scenario():
        async def refused(text, scope):
            return {"status": "error", "error": "no worker"}

        live = lc.HerLiveContext("s", read=refused)
        await live.open([_said("Hi.", "Hello.")])
        assert live.ready is False

    asyncio.run(scenario())


def test_the_worker_reads_into_the_session_scope_and_keeps_it(monkeypatch):
    from core.brain.llm import mlx_worker, mlx_worker_read_ahead

    calls = []
    held = [SimpleNamespace(state="layer state")]  # a cache entry, as mlx_lm's are: what it holds is its .state

    class Lru:
        def _holds(self, key, tokens):
            return False

        def fetch_nearest_cache(self, key, tokens, **_kw):
            calls.append(("fetch", key, list(tokens)))
            return held, tokens[2:]

        def insert_cache(self, key, tokens, cache):
            calls.append(("insert", key, list(tokens), cache))

    class Model:
        def __call__(self, ids, cache):
            calls.append(("read", ids))

    class Mx2:
        @staticmethod
        def array(xs):
            class Batch(list):
                def __getitem__(self, item):
                    return list(self)
            return Batch(xs)

        @staticmethod
        def eval(_x):
            return None

    monkeypatch.setattr(mlx_worker, "_runtime_prefill_step_size", lambda path: 2)
    sent = []
    tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3, 4, 5])
    import contextlib

    mlx_worker_read_ahead.read_ahead(
        {"id": "r1", "prompt": "some text", "prompt_cache_scope": "voice_live:s"}, model=Model(), tokenizer=tokenizer,
        model_path="/models/Her-27B", prompt_cache_lru=Lru(), ipc_writer=SimpleNamespace(put=sent.append),
        metal_semaphore=contextlib.nullcontext(), mx=Mx2())
    assert calls[0] == ("fetch", ("Her-27B", "voice_live:s"), [1, 2, 3, 4, 5])
    assert [c for c in calls if c[0] == "read"] == [("read", [3, 4]), ("read", [5])]
    assert calls[-1] == ("insert", ("Her-27B", "voice_live:s"), [1, 2, 3, 4, 5], held)
    assert sent[-1]["status"] == "ok" and sent[-1]["read"] == 3 and sent[-1]["id"] == "r1"


def test_text_already_held_is_not_read_twice(monkeypatch):
    from core.brain.llm import mlx_worker_read_ahead

    sent = []
    lru = SimpleNamespace(_holds=lambda key, tokens: True)
    mlx_worker_read_ahead.read_ahead(
        {"id": "r2", "prompt": "x", "prompt_cache_scope": "voice_live:s"}, model=None,
        tokenizer=SimpleNamespace(encode=lambda t: [9]), model_path="m", prompt_cache_lru=lru,
        ipc_writer=SimpleNamespace(put=sent.append), metal_semaphore=None, mx=None)
    assert sent[-1]["read"] == 0 and sent[-1]["status"] == "ok"


@pytest.mark.parametrize("scope, expected", [("voice_live:abc", "voice_live:abc"), ("other:abc", "default"),
                                             ("voice_live:" + "x" * 90, "default")])
def test_only_a_voice_session_names_its_own_scope(scope, expected):
    from core.brain.llm.mlx_worker import _prompt_cache_scope_for_job

    assert _prompt_cache_scope_for_job({"prompt_cache_scope": scope}) == expected


def test_a_lease_for_a_person_s_turn_is_a_foreground_lease(monkeypatch):
    import threading

    from core.brain import llm_health_router as router

    monkeypatch.setattr(router, "_GENERATION_GATE", threading.BoundedSemaphore(1))
    monkeypatch.setattr(router, "_GENERATION_GATE_ACTIVE_LEASES", {})
    monkeypatch.setattr(router, "_GENERATION_GATE_LEASE_DEADLINES", {})
    monkeypatch.setattr(router, "_GENERATION_GATE_FOREGROUND_LEASES", set())
    lease = asyncio.run(router.acquire_external_generation_gate_lease(owner="voice_live:reply", timeout_s=5,
                                                                      wait_s=0.1, foreground=True))
    assert lease in router._GENERATION_GATE_FOREGROUND_LEASES
    router.release_external_generation_gate_lease(lease)


def test_a_session_opens_on_this_sitting_and_the_last_exchange_not_every_exchange_there_is():
    """LIVE 11 November 2026: opening on the forty most recent exchanges read 12,326 tokens before her first turn."""
    import time

    now = time.time()
    earlier = [_said(f"Earlier question {n}?", "An earlier answer. " * 40, at=now - 2 * 86400 + n) for n in range(30)]
    this_sitting = [_said("Hey, are you there?", "I'm here.", at=now - 120), _said("Good.", "Good.", at=now - 60)]
    read = lc.what_she_reads_first([*earlier, *this_sitting], now=now)
    assert [pair["user"] for pair in read] == ["Hey, are you there?", "Good."]
    # After a break, the last exchange alone: a new turn answers into it.
    assert [pair["user"] for pair in lc.what_she_reads_first(earlier, now=now)] == ["Earlier question 29?"]
    # A sitting too long to read whole: the last exchange alone, as her governed turns read it.
    long_sitting = [_said(f"Question {n}?", "A long answer. " * 400, at=now - 600 + n) for n in range(5)]
    assert [pair["user"] for pair in lc.what_she_reads_first(long_sitting, now=now)] == ["Question 4?"]
    # What her runtime did not record as delivered is not read at all.
    assert lc.what_she_reads_first([{"user": "Hi.", "aura": "Hello.", "timestamp": now}], now=now) == []


def test_a_person_who_began_before_the_context_opened_still_has_their_turn_read_ahead(monkeypatch):
    """LIVE 11 November 2026: the person began 0.45 s before the context opened, and nothing of their turn was read
    while they spoke."""
    monkeypatch.setattr(lc, "the_turns_evidence", lambda: "[EVIDENCE]\n")

    async def scenario():
        live, read, generated, _recorded = _a_context()
        live.a_person_began_to_speak()  # before open: nothing to read into yet
        await live.open([_said("Hi.", "Hello.")])
        await asyncio.sleep(0)
        assert [text for text, _scope in read][-1].endswith("<|im_start|>user\n[EVIDENCE]\n")
        await live.answer("Are you there?")
        assert live._speaking is False

    asyncio.run(scenario())


def test_a_turn_that_ends_while_the_context_opens_waits_for_it():
    """LIVE 11 November 2026: the person stopped while it was opening, and her governed route took 16 s."""
    async def scenario():
        release = asyncio.Event()
        read, generated = [], []

        async def slow_read(text, scope):
            read.append(text)
            await release.wait()
            return {"status": "ok"}

        async def generate(prompt, **_kw):
            generated.append(prompt)
            return "I'm here."

        live = lc.HerLiveContext("s", read=slow_read, generate=generate)
        opening = asyncio.ensure_future(live.open([_said("Hi.", "Hello.")]))
        await asyncio.sleep(0)
        assert live.answers("Are you there?"), "an opening under way answers"
        answering = asyncio.ensure_future(live.answer("Are you there?"))
        await asyncio.sleep(0.01)
        assert not generated, "nothing is generated before the context is read"
        release.set()
        assert await answering == "I'm here."
        await opening
        assert generated[0].startswith(read[0])

    asyncio.run(scenario())


def test_a_turn_after_an_opening_that_failed_takes_her_governed_route():
    async def scenario():
        async def refused(text, scope):
            return {"status": "error"}

        live = lc.HerLiveContext("s", read=refused)
        await live.open([_said("Hi.", "Hello.")])
        assert not live.answers("Are you there?")
        assert await live.answer("Are you there?") is None

    asyncio.run(scenario())


def test_her_reply_asks_a_background_holder_to_stop_until_it_has_her_model(monkeypatch):
    """LIVE 11 November 2026: her narrative took the gate after the turn's one ask, and her reply waited 84 s."""
    from core.brain import llm_health_router as router
    from core.brain import llm_health_router_gate_owners as owners

    asked, attempts, released = [], [], []
    monkeypatch.setattr(owners, "_background_stands_aside", lambda reason: asked.append(reason))

    async def acquire(**kwargs):
        attempts.append(kwargs["wait_s"])
        return 7 if len(attempts) == 3 else None  # the holder lets go on the third ask

    monkeypatch.setattr(router, "acquire_external_generation_gate_lease", acquire)
    monkeypatch.setattr(router, "release_external_generation_gate_lease", released.append)

    class Client:
        async def generate(self, prompt, **kwargs):
            return "I'm here."

    monkeypatch.setattr(lc, "_her_cortex_client", Client)
    reply = asyncio.run(lc._generate_on_her_cortex("prompt", scope="voice_live:s", words="Hi?", reply_stream=None,
                                                   timeout_s=30.0))
    assert reply == "I'm here." and len(asked) == 3 and all(wait <= 1.0 for wait in attempts) and released == [7]


def test_a_reply_cut_off_while_it_was_written_goes_in_as_far_as_she_said_it(monkeypatch):
    """LIVE 11 November 2026: a reply cut off while it was written never went into her live context, and "Wait, stop.
    Just tell me in one sentence." reached her with nothing before it; she answered "I am an AI assistant here to help
    you with any questions or tasks you have.\""""
    monkeypatch.setattr(lc, "the_turns_evidence", lambda: "")

    async def scenario():
        started = asyncio.Event()

        async def a_long_reply(prompt, **_):
            started.set()
            await asyncio.sleep(30)

        live = lc.HerLiveContext("session-1", read=lambda *a: None, generate=a_long_reply)
        await live.open([])
        turn = asyncio.ensure_future(live.answer("Tell me about something you find beautiful."))
        await started.wait()
        live.cut_off("I find it beautiful when silence is active.")  # she got this far out loud
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn
        assert live.text.endswith("Tell me about something you find beautiful.<|im_end|>\n<|im_start|>assistant\n"
                                  "<think>\n\n</think>\n\nI find it beautiful when silence is active.<|im_end|>\n")
        live.cut_off("anything")  # once per reply
        assert live.text.endswith("silence is active.<|im_end|>\n")

    asyncio.run(scenario())


def test_a_whole_reply_cut_off_while_it_was_spoken_keeps_only_what_was_heard(monkeypatch):
    monkeypatch.setattr(lc, "the_turns_evidence", lambda: "")

    async def scenario():
        live, *_ = _a_context(replies=("First part. Second part never heard.",))
        await live.open([])
        await live.answer("Tell me two things.")
        live.cut_off("First part.")
        assert live.text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\nFirst part.<|im_end|>\n")
        live.heard_elsewhere("Open Safari.", "Opening it now, and then")
        live.cut_off("Opening it now,")
        assert live.text.endswith("<|im_start|>user\nOpen Safari.<|im_end|>\n<|im_start|>assistant\nOpening it now,<|im_end|>\n")

    asyncio.run(scenario())


def test_her_mind_bridge_says_what_she_got_out_loud_when_she_is_cut_off():
    from core.voice.duplex.mind_bridge import MindBridge, SpokenRecord

    told = []
    bridge = MindBridge(session_id="s", on_cut_off=told.append)
    bridge.record_spoken(SpokenRecord(intended="A. B.", spoken="A.", interrupted=True))
    bridge.record_spoken(SpokenRecord(intended="C.", spoken="C.", interrupted=False))
    assert told == ["A."]


def test_a_governed_reply_cut_off_before_it_returned_still_goes_in():
    """LIVE 11 November 2026: the first turn went to her governed route (her live context was not open yet), she was
    cut off before its reply returned, and the context never had the turn."""
    live, *_ = _a_context()
    live.ready = True
    live.governed_turn("Tell me about something you find beautiful.")
    live.cut_off("Silence that holds a question.")
    assert live.text.endswith("<|im_start|>user\nTell me about something you find beautiful.<|im_end|>\n"
                              "<|im_start|>assistant\nSilence that holds a question.<|im_end|>\n")
