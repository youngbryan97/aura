"""Her own echo canceller: WebRTC's AEC3 against the exact audio she sent to be played.

On a Mac the browser's echo cancellation is Apple's voice processing, and while a page holds the microphone with it,
macOS turns down and thins every other app's sound. LIVE 11 November 2026: Bryan's speakers sounded tinny in every
app while her app ran after a voice session, and normal again once he quit it. She knows what she played to the
sample, so she cancels it herself, and her window opens the microphone with no processing of its own.

Proved before it was wired (~/.aura/voice-lab/aec-2026-11-11/prove_aec.py): her own voice through a simulated room,
90 ms late, came out 40.6 dB quieter; a person talking over her came through 3 dB under their own level, about
38 dB above what was left of her.

Two things that proof did not have, both LIVE 11 November 2026 in her voice mode:

- Her speech is sent ahead of playing, by seconds when a sentence is made faster than real time, so it is placed
  where her window says it is playing (its playback reports, every 35 ms), not where it was sent.
- A microphone her voice does not reach (headphones; a stream with no room) leaves AEC3 nothing to learn, and it took
  the person's words for her echo: "Wait, stop. Just tell me in one sentence." said over her lost every voiced frame
  of its first 0.6 s and was heard as "Only in one oops." So whether her voice reaches the microphone is measured as
  she talks, and where it does not, the microphone passes with the rest of the processing but no echo suppression.
"""
from __future__ import annotations

import logging
import time
from collections import deque

import numpy as np

logger = logging.getLogger("Aura.Voice.Echo")

#: The rate her window's microphone reaches her at, and the one cancellation runs at.
RATE = 16000
#: WebRTC's audio processing works in 10 ms frames.
FRAME = RATE // 100
#: The last ten seconds of frames say whether her voice reaches the microphone; one second of each kind decides it.
_MEASURED, _ENOUGH = 1000, 100
#: Frames after her words that may still hold their echo: a room's and her window's output delay end within 300 ms.
_ECHO_REACH = 30


def cancels_acoustically() -> bool:
    """Whether her own canceller can run here: WebRTC's audio processing, as LiveKit builds it."""
    try:
        from livekit import rtc  # noqa: F401
    except ImportError:
        return False
    return True


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0


class HerEchoCanceller:
    """What she is playing, placed where her window plays it, and taken out of the microphone when it reaches it."""

    def __init__(self) -> None:
        from livekit import rtc

        self._rtc = rtc
        # What the browser did for her before: echo, low rumble, steady noise and level. Hers now, so macOS is never asked.
        self._apm = rtc.AudioProcessingModule(echo_cancellation=True, high_pass_filter=True, noise_suppression=True,
                                              auto_gain_control=True)
        # The same without echo suppression, for a microphone her voice does not reach.
        self._plain = rtc.AudioProcessingModule(echo_cancellation=False, high_pass_filter=True,
                                                noise_suppression=True, auto_gain_control=True)
        self._paced = np.zeros(0, dtype=np.float32)  # audio with no playback reports (asides): played from when sent
        self._sent = np.zeros(0, dtype=np.float32)  # this reply's speech as sent
        self._cursor = 0  # how much of it the canceller has been given
        self._gain = 1.0  # her window's duck
        self._report: tuple[float, float] | None = None  # (samples played, when her window said so)
        self._mic_rest = np.zeros(0, dtype=np.int16)
        self._while_playing: deque[float] = deque(maxlen=_MEASURED)
        self._while_quiet: deque[float] = deque(maxlen=_MEASURED)
        self.reaches_microphone = True  # until measured otherwise: her voice is cancelled
        self._since_her = _ECHO_REACH  # frames since her voice was last in what plays
        self._energy = [0.0, 0]  # her speech's sum of squares and samples: how loud she is

    def sent_to_be_played(self, samples: np.ndarray, rate: int, *, reported: bool = False) -> None:
        """Her speech, as it left for the speakers. ``reported``: her window will say where it is playing it."""
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if rate != RATE and audio.size:
            from math import gcd

            from scipy.signal import resample_poly

            divisor = gcd(int(rate), RATE)
            audio = resample_poly(audio, RATE // divisor, int(rate) // divisor).astype(np.float32)
        self._energy[0] += float(np.sum(np.square(audio, dtype=np.float64)))
        self._energy[1] += audio.size
        if reported:
            if self._report is None and audio.size:
                self._report = (float(self._sent.size), time.monotonic())  # until her window says: playing from now
            self._sent = np.concatenate([self._sent, audio])
        else:
            self._paced = np.concatenate([self._paced, audio])

    def new_reply(self) -> None:
        """A reply begins: its speech is placed from its own start, at her full level."""
        self._sent, self._cursor, self._report, self._gain = np.zeros(0, dtype=np.float32), 0, None, 1.0

    def ducked(self, gain: float) -> None:
        """Her window turned her down (someone began talking over her) or back up: what plays is at this gain."""
        self._gain = min(1.0, max(0.0, float(gain)))

    def played(self, seconds: float) -> None:
        """Her window's report: this much of the reply has played."""
        self._report = (max(0.0, float(seconds)) * RATE, time.monotonic())

    def stopped_playing(self) -> None:
        """Interrupted: what was still to play will not be heard."""
        self._paced = np.zeros(0, dtype=np.float32)
        self.new_reply()

    def _frame(self, pcm: np.ndarray):
        return self._rtc.AudioFrame(pcm.astype(np.int16).tobytes(), RATE, 1, FRAME)

    def _far(self, frames: int) -> np.ndarray:
        """What was playing while these microphone frames were heard: the reply where her window last said it was,
        moved on at real time since, and any aside paced from when it was sent."""
        span = frames * FRAME
        far = np.zeros(span, dtype=np.float32)
        if self._report is not None:
            played, said_at = self._report
            end = int(min(float(self._sent.size), played + (time.monotonic() - said_at) * RATE))
            first = max(self._cursor, end - span)  # never played twice; behind her window, it catches up
            piece = self._sent[first:end]
            far[first - (end - span):first - (end - span) + piece.size] += piece * self._gain
            self._cursor = max(self._cursor, end)
        aside, self._paced = self._paced[:span], self._paced[span:]
        far[:aside.size] += aside
        return far

    def _measure(self, far: np.ndarray, mic: np.ndarray) -> None:
        """Whether her voice reaches the microphone: when it does, the microphone's median level while she plays (and
        for 300 ms after, while her echo lasts) is at least twice its median in her silence: her echo at least as loud
        as the room. The median, not a floor: the first frames of every word arrive before their echo."""
        loud = (self._energy[0] / self._energy[1]) ** 0.5 if self._energy[1] else 0.0
        playing = _rms(far) >= 0.1 * loud if loud else False  # within 20 dB of her own level: her voice, not a pause
        self._since_her = 0 if playing else self._since_her + 1
        (self._while_playing if self._since_her < _ECHO_REACH else self._while_quiet).append(_rms(mic))
        if len(self._while_playing) < _ENOUGH or len(self._while_quiet) < _ENOUGH:
            return
        level_playing = float(np.median(self._while_playing))
        level_quiet = float(np.median(self._while_quiet))
        reaches = level_playing > 0.0 and level_playing >= 2.0 * level_quiet
        if reaches != self.reaches_microphone:
            self.reaches_microphone = reaches
            logger.info("🔇 her voice %s the microphone (%.1f dB over the room while she plays): %s",
                        "reaches" if reaches else "does not reach",
                        20.0 * np.log10((level_playing + 1e-9) / (level_quiet + 1e-9)),
                        "her echo is cancelled" if reaches else "words over her pass without echo suppression")

    def clean(self, data: bytes) -> bytes:
        """The microphone's PCM16 with her own voice taken out. A partial frame waits for the next buffer."""
        mic = np.concatenate([self._mic_rest, np.frombuffer(data, dtype=np.int16)])
        whole = (mic.size // FRAME) * FRAME
        self._mic_rest = mic[whole:].copy()
        out = np.empty(whole, dtype=np.int16)
        far_all = self._far(whole // FRAME)
        for start in range(0, whole, FRAME):
            far = far_all[start:start + FRAME]
            self._apm.process_reverse_stream(self._frame(np.clip(far, -1.0, 1.0) * 32767.0))
            heard = mic[start:start + FRAME]
            self._measure(far, heard.astype(np.float32) / 32768.0)
            near, plain = self._frame(heard), self._frame(heard)
            self._apm.process_stream(near)
            self._plain.process_stream(plain)
            out[start:start + FRAME] = np.frombuffer(bytes((near if self.reaches_microphone else plain).data),
                                                     dtype=np.int16)
        return out.tobytes()
