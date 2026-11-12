"""She takes her own voice out of the microphone, against the audio she sent to be played.

LIVE 11 November 2026: after a voice session Bryan's speakers sounded tinny in every app while her app ran, and normal
once he quit it. Her voice mode opened the microphone with the browser's echo cancellation, which on a Mac is Apple's
voice processing, and that turns down and thins all other sound. Her canceller is WebRTC's AEC3 (LiveKit's build); with
it, her window opens the microphone with no processing of its own.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("livekit")

from core.voice.duplex.echo_guard import EchoGuard  # noqa: E402
from core.voice.duplex.her_echo_canceller import FRAME, RATE, HerEchoCanceller  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]


def _speech_like(seconds: float, seed: int) -> np.ndarray:
    """Band-limited noise in syllable-length bursts: the shape of speech, from a seed."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    noise = rng.standard_normal(n).astype(np.float32)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1 / RATE)
    spectrum[(freqs < 150) | (freqs > 3800)] = 0
    voiced = np.fft.irfft(spectrum, n).astype(np.float32)
    syllables = (np.sin(2 * np.pi * 4.0 * np.arange(n) / RATE) > -0.2).astype(np.float32)
    out = voiced * syllables
    return (0.25 * out / (np.sqrt(np.mean(out ** 2)) + 1e-9)).astype(np.float32)


def _db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-12))


def _run(canceller: HerEchoCanceller, mic: np.ndarray, block: int = 640) -> np.ndarray:
    pcm = (np.clip(mic, -1, 1) * 32767).astype(np.int16).tobytes()
    out = b"".join(canceller.clean(pcm[i:i + block * 2]) for i in range(0, len(pcm), block * 2))
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32767


def _a_room(her: np.ndarray, delay_s: float, gain: float) -> np.ndarray:
    rng = np.random.default_rng(3)
    room = rng.standard_normal(int(0.08 * RATE)).astype(np.float32) * np.exp(-np.arange(int(0.08 * RATE)) / (0.02 * RATE))
    room /= np.sqrt(np.sum(room ** 2))
    room[0] += 1.0
    late = np.concatenate([np.zeros(int(delay_s * RATE), np.float32), her])
    return np.convolve(late, room)[: len(late)] * gain


def test_her_voice_coming_back_through_the_room_is_taken_out():
    canceller = HerEchoCanceller()
    her = _speech_like(5.0, seed=1)
    canceller.sent_to_be_played(her, RATE)
    mic = _a_room(her, delay_s=0.09, gain=0.5)
    out = _run(canceller, mic)
    settled = slice(int(2.0 * RATE), int(4.5 * RATE))
    assert _db(mic[settled]) - _db(out[settled]) > 20.0, "her echo comes out at least 20 dB quieter once it has learned the room"


def test_a_person_talking_over_her_still_comes_through():
    canceller = HerEchoCanceller()
    her = _speech_like(6.0, seed=1)
    person = np.zeros(len(her) + int(0.09 * RATE), np.float32)
    starts = int(3.5 * RATE)
    voice = _speech_like(2.0, seed=2)
    person[starts:starts + len(voice)] = voice
    canceller.sent_to_be_played(her, RATE)
    mic = _a_room(her, delay_s=0.09, gain=0.5) + person
    out = _run(canceller, mic)
    alone = slice(int(2.5 * RATE), starts)
    over = slice(starts, starts + len(voice))
    assert _db(out[over]) - _db(out[alone]) > 15.0, "an interruption stands well clear of what is left of her"


def test_her_speech_at_another_rate_is_taken_at_hers():
    canceller = HerEchoCanceller()
    canceller.sent_to_be_played(np.zeros(24000, np.float32), 24000)  # one second at her voice's rate
    assert canceller._paced.size == RATE


def test_an_interruption_drops_what_was_still_to_play():
    canceller = HerEchoCanceller()
    canceller.sent_to_be_played(_speech_like(1.0, seed=1), RATE)
    canceller.stopped_playing()
    assert canceller._paced.size == 0


def test_a_partial_frame_waits_for_the_next_buffer():
    canceller = HerEchoCanceller()
    first = canceller.clean(np.zeros(FRAME + 37, np.int16).tobytes())
    second = canceller.clean(np.zeros(FRAME - 37, np.int16).tobytes())
    assert len(first) == FRAME * 2 and len(second) == FRAME * 2


def test_the_guard_without_her_canceller_passes_the_microphone_through():
    guard = EchoGuard()
    data = np.arange(320, dtype=np.int16).tobytes()
    assert guard.cancels_acoustically is False and guard.clean(data) == data
    guard.note_spoken("hello there friend", np.zeros(10, np.float32), 24000)
    guard.stopped_playing()


def test_the_guard_with_her_canceller_says_so_and_feeds_it():
    guard = EchoGuard(acoustic=True)
    assert guard.cancels_acoustically is True
    guard.note_spoken("hello there friend", np.zeros(24000, np.float32), 24000)
    assert guard._acoustic._paced.size == RATE
    guard.stopped_playing()
    assert guard._acoustic._paced.size == 0


def test_her_window_turns_its_own_processing_off_only_when_she_cancels_echo_herself():
    source = (_ROOT / "interface/static/voice_mode.js").read_text(encoding="utf-8")
    assert "state.herOwnEchoCancellation = msg.echo_cancellation === 'hers';" in source
    for constraint in ("echoCancellation", "noiseSuppression", "autoGainControl"):
        assert re.search(rf"{constraint}: !state\.herOwnEchoCancellation", source), constraint
    session = (_ROOT / "core/voice/duplex/session.py").read_text(encoding="utf-8")
    assert '"echo_cancellation": "hers" if self._echo.cancels_acoustically else "browser"' in session
    assert "pcm16_to_float32(self._echo.clean(data))" in session


def test_her_speech_is_taken_out_where_her_window_says_it_is_playing(monkeypatch):
    """LIVE 11 November 2026: her speech is sent ahead of playing, by seconds when a sentence is made faster than real
    time; paced from when it was sent, the canceller's copy of her ran ahead of the room's."""
    import core.voice.duplex.her_echo_canceller as module

    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    canceller = HerEchoCanceller()
    her = np.arange(3 * RATE, dtype=np.float32) / (3 * RATE)
    canceller.sent_to_be_played(her, RATE, reported=True)  # three seconds sent at once
    canceller.played(1.0)  # her window: one second of it has played
    far = canceller._far(2)
    assert np.allclose(far, her[RATE - 2 * FRAME:RATE])
    canceller.played(1.0)  # the same report again (her window paused): nothing is played twice
    assert not canceller._far(2).any()
    now[0] += 0.5  # half a second later, no report: it moved on at real time
    assert np.allclose(canceller._far(1), her[int(1.5 * RATE) - FRAME:int(1.5 * RATE)])
    canceller.ducked(0.32)  # someone began talking over her, and her window turned her down
    now[0] += 0.01
    assert np.allclose(canceller._far(1), 0.32 * her[int(1.5 * RATE):int(1.5 * RATE) + FRAME])


def test_a_microphone_her_voice_does_not_reach_passes_words_over_her_whole():
    """LIVE 11 November 2026: with no echo to learn (headphones; a stream with no room) her canceller took the person
    for her echo, and "Wait, stop. Just tell me in one sentence." lost every voiced frame of its first 0.6 s."""
    canceller = HerEchoCanceller()
    her, person = _speech_like(6.0, seed=1), _speech_like(2.0, seed=2)
    quiet = np.zeros(2 * RATE, np.float32)
    canceller.sent_to_be_played(np.concatenate([np.zeros(2 * RATE, np.float32), her]), RATE)
    mic = np.concatenate([quiet, np.zeros(len(her), np.float32)]) + 0.001 * _speech_like(8.0, seed=4)  # the room
    starts = len(quiet) + int(4.0 * RATE)
    mic[starts:starts + len(person)] += person
    out = _run(canceller, mic)
    assert canceller.reaches_microphone is False
    over = slice(starts, starts + len(person))
    assert _db(mic[over]) - _db(out[over]) < 3.0, "the person comes through near their own level"


def test_her_voice_in_the_room_keeps_her_canceller_on():
    canceller = HerEchoCanceller()
    her = _speech_like(6.0, seed=1)
    canceller.sent_to_be_played(np.concatenate([np.zeros(2 * RATE, np.float32), her]), RATE)
    mic = _a_room(np.concatenate([np.zeros(2 * RATE, np.float32), her]), delay_s=0.09, gain=0.5)
    mic += 0.001 * _speech_like(len(mic) / RATE, seed=4)
    _run(canceller, mic)
    assert canceller.reaches_microphone is True
