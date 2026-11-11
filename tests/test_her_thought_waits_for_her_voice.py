"""Her thought waits for her voice: while a sentence of hers waits to be spoken, her model holds between tokens.

LIVE 11 November 2026: while her cortex wrote a long reply, her voice ran at 0.42 to 0.5 of real time on the GPU they
share and fell silent mid-reply for up to 15 s; with the GPU to itself it ran at 1.06 to 1.64
(core/brain/llm/speech_hold.py).
"""
from __future__ import annotations

import asyncio
import time

import pytest

from core.brain.llm import speech_hold
from core.conversation.reply_stream import ReplyStreamChannel


@pytest.fixture
def hold():
    owner = speech_hold.SpeechHold()
    yield owner
    owner.close()
    if speech_hold._READER is not None:
        speech_hold._READER.close()
        speech_hold._READER = None


def test_her_model_waits_while_held_and_goes_on_when_released(hold):
    job = {"speech_hold": hold.name}
    assert speech_hold.wait_for_her_voice(job) < 0.05  # nothing held
    hold.hold_for(0.3)
    waited = speech_hold.wait_for_her_voice(job)
    assert 0.25 <= waited < 0.6  # held, then lapsed on its own when nobody said it again


def test_a_hold_is_never_longer_than_half_her_heartbeats_stall_period(hold, monkeypatch):
    monkeypatch.setattr(speech_hold, "_longest_hold_s", lambda: 0.2)
    hold.hold_for(30.0)
    started = time.time()
    speech_hold.wait_for_her_voice({"speech_hold": hold.name})
    assert time.time() - started < 0.5
    from core.brain.llm.mlx_worker import HeartbeatThread

    monkeypatch.undo()
    assert speech_hold._longest_hold_s() == HeartbeatThread.LOOP_STALL_REPORT_S / 2


def test_a_job_with_no_hold_or_a_hold_already_gone_never_waits(hold):
    assert speech_hold.wait_for_her_voice({}) == 0.0
    assert speech_hold.wait_for_her_voice({"speech_hold": "aura-no-such-hold"}) == 0.0


def test_the_stream_says_how_much_is_waiting_to_be_spoken():
    async def scenario():
        channel = ReplyStreamChannel()
        assert channel.backlog() == 0
        channel.attach_loop(asyncio.get_running_loop())
        channel.publish("First sentence.")
        waiting_in_the_speaker = [2]
        channel.heard_backlog(lambda: waiting_in_the_speaker[0])
        assert channel.backlog() == 3
        waiting_in_the_speaker[0] = 0
        assert channel.backlog() == 1

    asyncio.run(scenario())


def test_her_thought_holds_while_a_sentence_waits_and_goes_on_when_her_voice_takes_it():
    from core.voice.duplex import live_context

    class Hold:
        def __init__(self):
            self.said = []

        def hold_for(self, seconds):
            self.said.append("hold")

        def release(self):
            self.said.append("go")

    class Stream:
        waiting = 1

        def backlog(self):
            return self.waiting

    async def scenario():
        hold, stream = Hold(), Stream()
        pacing = asyncio.ensure_future(live_context._her_thought_waits_for_her_voice(hold, stream))
        await asyncio.sleep(0.05)
        stream.waiting = 0
        await asyncio.sleep(0.15)
        pacing.cancel()
        return hold.said

    said = asyncio.run(scenario())
    assert said[0] == "hold" and said[-1] == "go"


def test_her_reply_is_owned_by_its_progress_not_by_a_clock(monkeypatch):
    from core.utils.deadlines import Deadline
    from core.voice.duplex import live_context

    seen = {}

    class Client:
        async def generate_text_to_completion(self, prompt, **kwargs):
            seen.update(kwargs, prompt=prompt)
            return "Four."

        async def generate(self, prompt, **kwargs):
            raise AssertionError("a reply held for her voice must not be cut at a clock set before it began")

    assert asyncio.run(live_context._her_reply(Client(), "p", {"speech_hold": "h"}, 90.0)) == "Four."
    assert isinstance(seen["deadline"], Deadline) and seen["speech_hold"] == "h" and seen["prompt"] == "p"


def test_her_worker_waits_where_it_sends_her_sentences(monkeypatch):
    from core.brain.llm import reply_previews

    asked = []
    monkeypatch.setattr(speech_hold, "wait_for_her_voice", lambda job: asked.append(job.get("speech_hold")) or 0.0)
    reply_previews.ready_reply_preview({"speech_hold": "h"}, "Hello. More", thinking=False, attempt=0)
    assert asked == ["h"]


@pytest.mark.asyncio
async def test_her_voice_says_it_is_making_speech_for_exactly_as_long_as_it_is(monkeypatch):
    """Her thought holds while her voice works, not only while a sentence waits: holding only then, her model wrote
    during her voice's work and her voice ran at 0.72 to 1.15 of real time (11 November 2026)."""
    import numpy as np

    from core.voice.duplex.config import TtsConfig
    from core.voice.duplex.prosody import ProsodySpec
    from core.voice.duplex.tts_stream import CancellationToken, StreamingTts

    tts = StreamingTts(TtsConfig(treatment="clean"))
    seen = []

    def synthesize_pieces(text, spec, deliver, stopped=None):
        seen.append(tts.making_speech())
        deliver(np.ones(480, np.float32), 24000, False)

    tts._state.pocket = type("Voice", (), {"name": "voxcpm2", "streams": True,
                                          "synthesize_pieces": staticmethod(synthesize_pieces)})()
    tts._state.loaded = True
    tts._accepting_synthesis = True
    monkeypatch.setattr(tts, "ensure_loaded", lambda: asyncio.sleep(0, result=True))
    try:
        assert tts.making_speech() == 0
        results = [r async for r in tts.synthesize_as_it_forms("A clause.", ProsodySpec(voice="aura_kiwi"),
                                                               CancellationToken())]
        assert seen == [1] and len(results) == 1
        assert tts.making_speech() == 0
    finally:
        tts.shutdown()


def test_her_thought_is_released_early_enough_for_her_next_clause_to_be_heard_on_time():
    """LIVE 11 November 2026: released only once her voice had nothing left to make, a long reply went silent between
    sentences for up to 6.0 s. She is released once the audio queued is no longer than her measured lead."""
    from core.voice.duplex.her_turn_taking import TurnTaking

    now = [0.0]
    taking = TurnTaking(clock=lambda: now[0])
    assert taking.backlog(waiting=1, making=1, ahead_s=9.0) == 1  # a clause waits: hold
    assert taking.backlog(waiting=0, making=1, ahead_s=9.0) == 1  # nothing learned yet: held while her voice works
    assert taking.backlog(waiting=0, making=0, ahead_s=1.0) == 0  # her voice has nothing to make: released
    now[0] = 2.5
    taking.first_sound_of_a_clause()  # her next clause was heard 2.5 s after the release
    assert taking.lead_s == 2.5
    assert taking.backlog(waiting=0, making=1, ahead_s=6.0) == 1  # 6 s queued covers her lead: hold
    assert taking.backlog(waiting=0, making=1, ahead_s=2.4) == 0  # no longer does: released in time
    now[0] = 4.0
    taking.first_sound_of_a_clause()
    assert taking.lead_s == pytest.approx(0.7 * 2.5 + 0.3 * 1.5)
    assert taking.backlog(waiting=0, making=0, ahead_s=20.0) == 0  # her voice has nothing to make: released
