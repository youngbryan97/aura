"""Her thought waits for her voice: while her voice is making speech or has her next sentence in hand, her model holds.

Her reply's text comes about four times faster than she can speak it (15 tokens a second against about 3.6), and her
voice model runs on the GPU her cortex is decoding on. LIVE 11 November 2026, while her cortex wrote a long reply her
voice made speech at 0.42 to 0.5 of real time and fell silent mid-reply for up to 15 s; with her GPU to itself it ran
at 1.06 to 1.64. So while her voice is making speech, or has her next sentence in hand, her model waits between
tokens; when her voice runs out of text it goes on. Her voice makes a sentence faster than it plays, so what it has
made covers her model writing the next one, and the two take turns on the GPU instead of sharing it. (Holding only
while a sentence waited, her model wrote during her voice's work, and her voice still ran at 0.72 to 1.15.)

The signal is one number in shared memory, "hold until" in seconds since the epoch: her voice's side writes it, her
worker reads it between tokens. It lapses by itself if her voice's side stops writing it, and a single hold is never
longer than half the period after which her worker's heartbeat reports a stalled job (mlx_worker.HeartbeatThread), so
a held generation still shows progress.
"""
from __future__ import annotations

import struct
import time
from multiprocessing import shared_memory

_UNTIL = struct.Struct("d")


class SpeechHold:
    """Her voice's side: owns the shared number and says how long to hold."""

    def __init__(self) -> None:
        self._memory = shared_memory.SharedMemory(create=True, size=_UNTIL.size)
        self.name = self._memory.name
        self.release()

    def hold_for(self, seconds: float) -> None:
        _UNTIL.pack_into(self._memory.buf, 0, time.time() + max(0.0, float(seconds)))

    def release(self) -> None:
        _UNTIL.pack_into(self._memory.buf, 0, 0.0)

    def close(self) -> None:
        try:
            self.release()
        finally:
            self._memory.close()
            try:
                self._memory.unlink()
            except FileNotFoundError:
                pass


class _Reader:
    def __init__(self, name: str) -> None:
        self.name = name
        self._memory = shared_memory.SharedMemory(name=name)
        try:  # the owner unlinks it; a reader that registered it would unlink it again at exit
            from multiprocessing import resource_tracker

            resource_tracker.unregister(self._memory._name, "shared_memory")  # noqa: SLF001
        except (ImportError, AttributeError, KeyError, ValueError):
            pass

    def until(self) -> float:
        return float(_UNTIL.unpack_from(self._memory.buf, 0)[0])

    def close(self) -> None:
        self._memory.close()


_READER: _Reader | None = None


def _longest_hold_s() -> float:
    try:
        from core.brain.llm.mlx_worker import HeartbeatThread

        return float(HeartbeatThread.LOOP_STALL_REPORT_S) / 2.0
    except (ImportError, AttributeError, TypeError, ValueError):
        return 15.0  # half of the heartbeat's 30 s, as it stood on 11 November 2026


def wait_for_her_voice(job: dict, *, sleep_s: float = 0.02) -> float:
    """Between tokens of a job that names a hold: wait while her voice has her next sentence in hand. Seconds waited."""
    global _READER
    name = str(job.get("speech_hold") or "") if isinstance(job, dict) else ""
    if not name:
        return 0.0
    if _READER is None or _READER.name != name:
        if _READER is not None:
            _READER.close()
            _READER = None
        try:
            _READER = _Reader(name)
        except (FileNotFoundError, OSError, ValueError):
            return 0.0  # her voice's side has already gone
    started = time.time()
    longest = _longest_hold_s()
    while True:
        now = time.time()
        try:
            until = _READER.until()
        except (ValueError, OSError):
            return now - started
        if until <= now or now - started >= longest:
            return now - started
        time.sleep(min(sleep_s, until - now))
