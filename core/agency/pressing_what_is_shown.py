"""Keys a screen shows during play as pictures, pressed while it shows them: in turn, and as fast as she can.

A screen that puts a key on itself mid-play is asking for it now. LIVE 2026-10-08
a game caught her character and showed ← and → over it, lit by turns: tap
them, one after the other, fast, to break free. She went on steering a character
that could not move, and lost.

A key drawn on the screen (core/perception/keys_drawn_on_screen.py) is asked
for when it appears: one that was not on the screen at some earlier look. Keys
shown at every look since play began are part of the screen's furniture (a
panel of the controls) and ask for nothing. While a key is asked for, everything
else waits: what she is holding is let go, and each key shown is tapped once a
picture, in the order they stand, which alternates them where there are two. A
person does the same with a prompt to mash a button, a quick-time event, or a
tutorial that lights the key to press next.
"""
from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("Aura.PressingWhatIsShown")

__all__ = ["KeysShown"]

#: How often she looks for keys drawn on the screen, in seconds: a look takes about a seventh of a second
#: (measured on a 868x588 picture), so she looks as often as the looking allows.
LOOK_EVERY_S = 0.15

#: How long each press is held, in seconds: longer than one frame at 24 frames a second (42 ms), the slowest a Flash
#: game runs, so a world that looks at its keys once a frame sees every press. A key pressed and let go in the same
#: instant is never seen held by such a world (measured on tests/fixtures/worlds_that_move/held.html: no press counted).
PRESSED_FOR_S = 0.06

#: Presses made before what is shown is said: by then the keys have been seen over a few looks.
SAID_AFTER = 4

#: How long a key last seen is still taken as shown, in seconds: two looks, so a key lit by turns, and missed at
#: one look while it is dark, is not dropped between them.
SHOWN_FOR_S = 2 * LOOK_EVERY_S + 0.1


class KeysShown:
    """What keys the screen shows in play, which it shows now that it did not always show, and pressing them."""

    def __init__(self, never_absent: set[str] | None = None) -> None:
        #: Keys shown at every look so far (None before the first): the screen's furniture.
        self.never_absent: set[str] | None = set(never_absent) if never_absent is not None else None
        self.asked: tuple[str, ...] = ()
        self.asked_at = -math.inf
        self.looking: asyncio.Future | None = None
        self.looked_at = -math.inf
        self.pressed = 0
        #: Each key asked for lately: when it was last seen asked, and where across the screen it stands.
        self.lately: dict[str, tuple[float, float]] = {}

    def look(self, picture: Any, at: float) -> None:
        """Take in the last look's keys if it is done, and start another if it is time (off the loop)."""
        from core.perception.keys_drawn_on_screen import keys_drawn

        if self.looking is not None and self.looking.done():
            drawn, when = self.looking.result()
            self.looking = None
            self._saw(drawn, when)
        if self.looking is None and at - self.looked_at >= LOOK_EVERY_S:
            self.looked_at = at
            copy = picture.copy()

            async def _look() -> tuple[list[dict[str, Any]], float]:
                return await asyncio.to_thread(keys_drawn, copy), at

            self.looking = asyncio.ensure_future(_look())

    def _saw(self, drawn: list[dict[str, Any]], when: float) -> None:
        shown = list(dict.fromkeys(str(key["key"]) for key in drawn))
        across = {str(key["key"]): float(key.get("center_x", 0.0)) for key in drawn}
        before = self.never_absent
        self.never_absent = set(shown) if before is None else before & set(shown)
        for key in shown:
            if before is not None and key not in before:
                self.lately[key] = (when, across[key])
        # The keys asked for are those seen asked in the last looks, not only the last: a key lit by turns, or read
        # at one look and not the next, is still being asked for (offline 2026-10-09 one of two was read at the first
        # look, and pressing it alone freed nothing).
        asked = tuple(key for key, (seen, _x) in sorted(self.lately.items(), key=lambda item: item[1][1])
                      if when - seen <= SHOWN_FOR_S)
        if asked:
            if asked != self.asked:
                logger.info("the screen shows %s, not shown at every look before: asked for now", ", ".join(asked))
            self.asked, self.asked_at = asked, max(seen for seen, _x in self.lately.values())

    def asks(self, at: float) -> bool:
        """Whether keys are asked for now."""
        return bool(self.asked) and at - self.asked_at <= SHOWN_FOR_S

    async def press(self, hands: Any, say: Callable[[str], Any] | None = None) -> None:
        """Each key asked for pressed once, in the order shown; said the first time."""
        if not self.asked:
            return
        # Said once the keys have been seen over a few looks, so a key read at one look and not yet at the next is
        # not left out of what she says.
        if say is not None and self.pressed == SAID_AFTER:
            keys = " and ".join(self.asked)
            say(f"It's showing {keys}: pressing {'them in turn' if len(self.asked) > 1 else 'it'}, fast.")
        for key in self.asked:
            await hands.down(key)
            await asyncio.sleep(PRESSED_FOR_S)
            await hands.up(key)
            self.pressed += 1

    def done(self) -> set[str] | None:
        """Let go of a look in flight; what was never absent, for the next stretch of the same game (None: not looked)."""
        if self.looking is not None:
            self.looking.cancel()
            self.looking = None
        return None if self.never_absent is None else set(self.never_absent)
