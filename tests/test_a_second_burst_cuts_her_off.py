"""A short burst over her voice is an acknowledgment only when quiet follows it.

LIVE 11 November 2026: "Wait, stop. Just tell me in one sentence." said over her was judged two acknowledgments (the
comma pause fell at the 460 ms decision, and the check of the words came back empty), and she talked on for 157 s.
"""
import asyncio
from types import SimpleNamespace

import numpy as np

from core.voice.duplex.session import SECOND_BURST_S, DuplexVoiceSession, _SpeakingTrack

SPEECH, QUIET = SimpleNamespace(probability=0.95), SimpleNamespace(probability=0.0)


class _SlowAsr:
    async def probe(self, _audio):
        await asyncio.sleep(30)  # the check of the words has not come back
        return ""

    def reset(self):
        return None

    def shutdown(self):
        return None


async def _burst(session, frames_of_speech: int, frames_quiet: int) -> None:
    frame = np.full(512, 0.1, dtype=np.float32)
    for _ in range(frames_of_speech):
        await session._handle_overlap(frame, SPEECH)
    for _ in range(frames_quiet):
        await session._handle_overlap(frame, QUIET)


def _speaking_session(turns: list) -> DuplexVoiceSession:
    async def send(_payload):
        return None

    session = DuplexVoiceSession(session_id="second-burst", send_json=send, send_binary=send)
    session._asr = _SlowAsr()
    session._speaking = _SpeakingTrack(utterance_id=1, intended="A long answer.", started_at=0.0)
    session._state = type(session._state).SPEAKING

    async def begin(captured_audio=None, reason=""):
        turns.append((reason, captured_audio.size))

    session._begin_user_turn = begin
    return session


def test_a_second_burst_before_the_first_was_heard_cuts_her_off():
    async def exercise():
        turns: list = []
        session = _speaking_session(turns)
        await _burst(session, 8, 7)  # "Wait," then the comma pause: a short burst, decided at 480 ms
        assert turns == [] and session._unheard_burst is not None
        await _burst(session, 8, 7)  # "stop."
        assert turns == [("user_barge_in", 2 * 15 * 512)]  # both bursts reach her turn
        await session.close()

    asyncio.run(exercise())


def test_one_burst_then_quiet_is_still_an_acknowledgment():
    async def exercise():
        turns: list = []
        session = _speaking_session(turns)
        await _burst(session, 8, 7)
        session._unheard_burst = (session._unheard_burst[0] - SECOND_BURST_S, session._unheard_burst[1])
        await _burst(session, 8, 7)  # a second "mhm", long after the first
        assert turns == []
        await session.close()

    asyncio.run(exercise())


def test_a_cut_off_turn_is_waited_for_as_long_as_her_client_waits_for_its_cancel(monkeypatch):
    """LIVE 11 November 2026: the turn a barge-in cut off took longer than 3 s to end (her client waits up to 12 s for
    her worker to acknowledge a cancel), and the session closed her voice mode instead of waiting for it."""
    import core.voice.duplex.session as session_module

    monkeypatch.setattr(session_module, "TASK_QUIESCENCE_TIMEOUT_S", 0.05)
    monkeypatch.setenv("AURA_MLX_SOFT_CANCEL_ACK_WAIT_S", "0.5")

    async def exercise():
        session = _speaking_session([])

        async def a_reply_whose_cancel_is_acknowledged_late():
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                await asyncio.shield(asyncio.sleep(0.2))  # as mlx_client waits for its worker
                raise

        session._turn_task = asyncio.ensure_future(a_reply_whose_cancel_is_acknowledged_late())
        await asyncio.sleep(0)
        session._speaking = None
        await session._cancel_active_turn(reason="user_barge_in", return_to_listening=True)
        assert session._turn_task is None and session.state.value == "listening"
        await session.close()

    asyncio.run(exercise())
