"""Her own echo canceller: WebRTC's AEC3 against the exact audio she sent to be played.

On a Mac the browser's echo cancellation is Apple's voice processing, and while a page holds the microphone with it,
macOS turns down and thins every other app's sound. LIVE 11 November 2026: Bryan's speakers sounded tinny in every
app while her app ran after a voice session, and normal again once he quit it. She knows what she played to the
sample, so she cancels it herself, and her window opens the microphone with no processing of its own.

Proved before it was wired (~/.aura/voice-lab/aec-2026-11-11/prove_aec.py): her own voice through a simulated room,
90 ms late, came out 40.6 dB quieter; a person talking over her came through 3 dB under their own level, about
38 dB above what was left of her.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("Aura.Voice.Echo")

#: The rate her window's microphone reaches her at, and the one cancellation runs at.
RATE = 16000
#: WebRTC's audio processing works in 10 ms frames.
FRAME = RATE // 100


def cancels_acoustically() -> bool:
    """Whether her own canceller can run here: WebRTC's audio processing, as LiveKit builds it."""
    try:
        from livekit import rtc  # noqa: F401
    except ImportError:
        return False
    return True


class HerEchoCanceller:
    """What she sent to be played, paced out at the rate the microphone's audio arrives, and taken out of it."""

    def __init__(self) -> None:
        from livekit import rtc

        self._rtc = rtc
        # What the browser did for her before: echo, low rumble, steady noise and level. Hers now, so macOS is never asked.
        self._apm = rtc.AudioProcessingModule(echo_cancellation=True, high_pass_filter=True, noise_suppression=True,
                                              auto_gain_control=True)
        self._playing = np.zeros(0, dtype=np.float32)
        self._mic_rest = np.zeros(0, dtype=np.int16)

    def sent_to_be_played(self, samples: np.ndarray, rate: int) -> None:
        """Her speech, as it left for the speakers. It is played from about now, at real time."""
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if rate != RATE and audio.size:
            from math import gcd

            from scipy.signal import resample_poly

            divisor = gcd(int(rate), RATE)
            audio = resample_poly(audio, RATE // divisor, int(rate) // divisor).astype(np.float32)
        self._playing = np.concatenate([self._playing, audio])

    def stopped_playing(self) -> None:
        """Interrupted: what was still to play will not be heard."""
        self._playing = np.zeros(0, dtype=np.float32)

    def _frame(self, pcm: np.ndarray):
        return self._rtc.AudioFrame(pcm.astype(np.int16).tobytes(), RATE, 1, FRAME)

    def clean(self, data: bytes) -> bytes:
        """The microphone's PCM16 with her own voice taken out. A partial frame waits for the next buffer."""
        mic = np.concatenate([self._mic_rest, np.frombuffer(data, dtype=np.int16)])
        whole = (mic.size // FRAME) * FRAME
        self._mic_rest = mic[whole:].copy()
        out = np.empty(whole, dtype=np.int16)
        for start in range(0, whole, FRAME):
            far, self._playing = self._playing[:FRAME], self._playing[FRAME:]
            if far.size < FRAME:
                far = np.concatenate([far, np.zeros(FRAME - far.size, dtype=np.float32)])
            self._apm.process_reverse_stream(self._frame(np.clip(far, -1.0, 1.0) * 32767.0))
            near = self._frame(mic[start:start + FRAME])
            self._apm.process_stream(near)
            out[start:start + FRAME] = np.frombuffer(bytes(near.data), dtype=np.int16)
        return out.tobytes()
