"""A locked screen is a reason not to look, known before looking.

LIVE 2026-09-23, all evening on a locked Mac: each ambient tick asked which
window was in front, waited out the whole twenty seconds, and after four in a
row the organ was reported as failing.
"""

from __future__ import annotations

import asyncio

from core.perception import ambient_presence


def test_a_locked_screen_skips_before_asking_what_is_in_front(monkeypatch):
    presence = ambient_presence.AmbientPresence()
    monkeypatch.setattr(ambient_presence, "_proactivity_suppressed", lambda: False)
    monkeypatch.setattr(ambient_presence, "_the_screen_is_locked", lambda: True)

    async def never(self):
        raise AssertionError("asked what is in front of a locked screen")

    monkeypatch.setattr(ambient_presence.AmbientPresence, "_current_context", never)
    result = asyncio.run(presence.tick())
    assert result.observed is False
    assert result.skip_reason is ambient_presence.SkipReason.SESSION_LOCKED
