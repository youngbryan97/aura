"""When her thought goes back to writing while her voice speaks: early enough that her next clause is heard on time.

Her thought holds while her voice is making speech (core/brain/llm/speech_hold.py), so the two take turns on the GPU.
LIVE 11 November 2026, released only once her voice had nothing left to make, a long reply still went silent between
sentences: once 4.1 s, once 6.0 s, and a dozen times under 0.2 s, because writing the next clause and making its first
sound took longer than the audio already queued. So her thought is released once the audio queued ahead of the
listener is no longer than her lead: how long, measured, it takes from releasing her thought to her next clause's
first sound.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


class TurnTaking:
    """Her lead, measured turn by turn, and whether her thought holds now."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.lead_s = 0.0
        self._clock = clock
        self._released_at: float | None = None

    def backlog(self, waiting: int, making: int, ahead_s: float) -> int:
        """What holds her thought: a clause waiting to be spoken, or her voice at work with more queued than her
        lead."""
        if waiting > 0 or (making > 0 and ahead_s > self.lead_s):
            self._released_at = None
            return max(1, waiting)
        if self._released_at is None:
            self._released_at = self._clock()
        return 0

    def hears(self, channel: Any, bridge: Any, session: Any) -> TurnTaking:
        """Say through ``channel`` what holds her thought: ``bridge``'s waiting clauses, her voice at work, and how much
        of what was sent the listener has still to hear (``session``'s track and the client's playback reports)."""

        def backlog() -> int:
            track = session._speaking
            ahead = track.sent_duration_s - session._client_played_s if track is not None else 0.0
            return self.backlog(bridge.qsize(), int(getattr(session._tts, "making_speech", int)()), ahead)

        channel.heard_backlog(backlog)
        return self

    def first_sound_of_a_clause(self) -> None:
        """A new clause was heard: the time since her thought was released is her lead, weighted to the last few."""
        if self._released_at is None:
            return
        sample = self._clock() - self._released_at
        self.lead_s = sample if not self.lead_s else 0.7 * self.lead_s + 0.3 * sample
        self._released_at = None
