"""The screen pursuit's reflexes: what moves on its own is played as it happens.

The screen pursuit reads a screen, decides, acts once and reads again. On a
menu, an instruction screen or a board that is the right pace. Once a game is
running it is not: LIVE 2026-10-03 she played Flash games on her own page at
one move every ten to fifteen seconds while the game went on without her.

So for the length of a hand-over a second, faster way of playing sits under
the first. Before each of the pursuit's looks it checks whether the picture is
moving on its own. If it is, it plays that stretch as it happens
(core/agency/playing_as_it_happens.py), with the controls the game's own words
named, and gives the pursuit back the screen it ends on: a game over, a level
done, a menu. The pursuit reads that the way it reads any screen.

A run is over, for whoever handed the game over, when she has played some of
it and a way to start again has appeared that was not there at the start. What
to do then (play again, go back to the list) belongs to the request, which the
caller holds.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.ScreenPursuit.AsItHappens")

__all__ = ["AS_IT_HAPPENS", "PlayingAsItHappens"]

#: How long the reflexes leave a moving screen alone after finding nothing on
#: it that answers to her: a title screen that animates is not a game yet.
LEAVE_A_MOVING_MENU_S = 20.0

#: The longest one stretch of play runs before the pursuit gets a look.
STRETCH_S = 300.0


@dataclass
class PlayingAsItHappens:
    """Eyes and hands on the part of her own page that draws, at the speed it goes."""

    page: Any
    band: tuple[float, float, float, float]
    goal: str
    ends_at: float
    keep: dict[str, Any] = field(default_factory=dict)
    stretches: list[dict[str, Any]] = field(default_factory=list)
    words: list[str] = field(default_factory=list)
    first_ways_back: frozenset[str] | None = None
    quiet_until: float = 0.0
    over_because: str = ""
    recalled: bool = False
    #: The words of the screen that ended the run, for whoever asks how it ended.
    ending_words: str = ""
    _clip: dict[str, float] | None = None
    _focused: bool = False

    # -- eyes ---------------------------------------------------------------

    async def _where(self) -> dict[str, float]:
        if self._clip is None:
            from core.perception.what_her_page_shows import the_page_size

            wide, tall = await the_page_size(self.page)
            left, top, right, bottom = self.band
            self._clip = {
                "x": left * wide, "y": top * tall,
                "width": max(1.0, (right - left) * wide), "height": max(1.0, (bottom - top) * tall),
            }
        return self._clip

    async def look(self) -> tuple[Any, float] | None:
        import cv2
        import numpy as np

        clip = await self._where()
        try:
            data = await self.page.screenshot(clip=clip, type="jpeg", quality=80)
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            return None
        at = time.monotonic()
        picture = await asyncio.to_thread(cv2.imdecode, np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if picture is None:
            return None
        return picture[:, :, ::-1], at

    # -- hands --------------------------------------------------------------

    async def _focus(self) -> None:
        if self._focused:
            return
        from core.skills.screen_pursuit_on_a_page import _FOCUS_THE_DRAWING

        try:
            self._focused = bool(await self.page.evaluate(_FOCUS_THE_DRAWING))
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            self._focused = False

    @staticmethod
    def _key(name: str) -> str:
        from core.skills.screen_pursuit_on_a_page import _KEY_NAMES

        return _KEY_NAMES.get(name, name)

    async def down(self, key: str) -> None:
        await self._focus()
        await self.page.keyboard.down(self._key(key))

    async def up(self, key: str) -> None:
        await self.page.keyboard.up(self._key(key))

    async def tap(self, key: str) -> None:
        await self._focus()
        await self.page.keyboard.press(self._key(key))

    async def _at(self, x: float, y: float) -> tuple[float, float]:
        clip = await self._where()
        return clip["x"] + min(1.0, max(0.0, x)) * clip["width"], clip["y"] + min(1.0, max(0.0, y)) * clip["height"]

    async def click(self, x: float, y: float) -> None:
        await self.page.mouse.click(*await self._at(x, y))
        self._focused = True

    async def point(self, x: float, y: float) -> None:
        await self.page.mouse.move(*await self._at(x, y))

    # -- the pursuit's hooks ---------------------------------------------------

    def read(self, observation: dict[str, Any]) -> None:
        """What a look of the pursuit showed: the screen's words, and its ways back at the start."""
        from core.skills.screen_pursuit_bearings import restart_controls

        said = " ".join(str(observation.get("text") or "").split())
        if said and (not self.words or self.words[-1] != said):
            self.words.append(said)
            del self.words[:-12]
        if self.first_ways_back is None and observation.get("ok", True):
            self.first_ways_back = restart_controls(observation)

    def run_is_over(self, observation: dict[str, Any]) -> bool:
        """Whether she has played, and the screen now offers a way to start again it did not offer at first."""
        if self.over_because:
            return True
        if not any(stretch.get("pictures") for stretch in self.stretches):
            return False
        from core.skills.screen_pursuit_bearings import restart_controls

        # Against what was on screen while she played, where she read it during
        # play: the screen that ended the last run already offered a way back,
        # so the first reading of this one is no measure of what is new.
        during = {said for stretch in self.stretches for said in stretch.get("words_seen") or ()}
        before = during if during else (self.first_ways_back or frozenset())
        appeared = {control for control in restart_controls(observation) if control not in before}
        if appeared:
            self.over_because = f"a way to start again appeared ({', '.join(sorted(appeared))})"
            self.ending_words = " ".join(str(observation.get("text") or "").split())
            logger.info("the run is over: %s", self.over_because)
            return True
        return False

    async def while_it_moves(self) -> None:
        """Play whatever is moving on its own, until it stops moving."""
        from core.agency.playing_as_it_happens import (
            controls_named_in,
            play_as_it_happens,
            the_world_moves_on_its_own,
        )
        from core.perception.what_the_pixels_show import recognize_text

        now = time.monotonic()
        if now < self.quiet_until or now >= self.ends_at:
            return
        if not await the_world_moves_on_its_own(self.look):
            return
        if not self.keep and not self.recalled:
            self.recalled = True
            self.keep.update(_what_she_kept_of(self.page))
        keys, pointer_first = controls_named_in(" ".join([self.goal, *self.words[-6:]]))
        logger.info("it moves on its own: playing it as it happens with %s%s", keys, " and the pointer" if pointer_first else "")
        stretch = await play_as_it_happens(
            self.look, self, keys=keys, seconds=min(STRETCH_S, self.ends_at - now),
            say=_said_while_playing, read_words=recognize_text, keep=self.keep,
            pointer_first=pointer_first, getting_somewhere=_getting_somewhere,
            told=" ".join([self.goal, *self.words[-6:]]),
        )
        self.stretches.append(stretch)
        _keep_what_she_learned(self.page, self.keep)
        logger.info("a stretch played as it happened: %s", {k: stretch.get(k) for k in ("seconds", "ended", "hers", "learned", "gains", "losses")})
        if not stretch.get("hers") and not stretch.get("gains") and not stretch.get("losses"):
            self.quiet_until = time.monotonic() + LEAVE_A_MOVING_MENU_S

    def what_it_came_to(self) -> str:
        """One line on the play, for whoever handed the game over."""
        played = [s for s in self.stretches if s.get("pictures")]
        if not played:
            return ""
        seconds = sum(float(s.get("seconds") or 0.0) for s in played)
        last = played[-1]
        parts = [f"played it as it happened for {seconds:.0f} s"]
        if last.get("hers"):
            parts.append(f"I was the {last['hers']}")
        learned = last.get("learned") or {}
        if learned:
            parts.append("; ".join(f"{colour}: {stance}" for colour, stance in learned.items()))
        counters = last.get("counters") or {}
        if counters:
            parts.append(", ".join(f"{name} {value}" for name, value in counters.items() if not name.startswith("number")))
        if self.over_because:
            parts.append(f"the run is over: {self.over_because}")
        return "; ".join(part for part in parts if part)


def _this_game(page: Any) -> str:
    """The name her memory keeps this game under: its page, and only its page."""
    return f"played as it happens at {str(getattr(page, 'url', '') or '')}"


def _what_she_kept_of(page: Any) -> dict[str, Any]:
    from core.agency.what_she_keeps_of_a_game import kept_from
    from core.runtime.what_she_learned import recall

    held = recall(_this_game(page))
    if held:
        logger.info("she has played this game before: starting from what she kept of it")
    return kept_from(held)


def _keep_what_she_learned(page: Any, keep: dict[str, Any]) -> None:
    from core.agency.what_she_keeps_of_a_game import to_keep
    from core.runtime.what_she_learned import remember

    remember(_this_game(page), to_keep(keep))


def _said_while_playing(line: str) -> None:
    from core.skills.screen_pursuit import _tell

    _tell(line)


def _getting_somewhere(said: str) -> None:
    from core.runtime.still_getting_somewhere import it_got_somewhere
    from core.skills.sovereign_browser_one_question import SAYING_IT_MOVES

    (SAYING_IT_MOVES.get() or it_got_somewhere)(said)


#: The reflexes for the hand-over in progress, when there is one.
AS_IT_HAPPENS: contextvars.ContextVar[PlayingAsItHappens | None] = contextvars.ContextVar(
    "aura_playing_as_it_happens", default=None
)
