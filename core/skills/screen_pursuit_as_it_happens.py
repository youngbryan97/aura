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

import contextvars
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.ScreenPursuit.AsItHappens")

__all__ = ["AS_IT_HAPPENS", "PlayingAsItHappens", "a_handed_over_run_is_over", "looked_at_as_it_happens"]


def begin_run(keep: dict[str, Any]) -> None:
    """A new attempt keeps world knowledge and measures its own state."""
    from core.agency.how_the_contest_stands import ContestStands
    from core.agency.which_one_answers_to_her import WhichIsHers

    contest = keep.get("contest")
    keep["contest"] = ContestStands(wins=contest.wins) if contest is not None else ContestStands()
    keep["hers"] = WhichIsHers()
    meeting = keep.get("meeting")
    if meeting is not None:
        meeting.begin_run()

#: How long the reflexes leave a moving screen alone after finding nothing on
#: it that answers to her: a title screen that animates is not a game yet.
LEAVE_A_MOVING_MENU_S = 20.0

#: The longest one stretch of play runs before the pursuit gets a look.
STRETCH_S = 300.0

#: How long she tries out a thing whose own words set her to make something, not to win: it has no end of its own.
TRYING_OUT_S = 90.0

#: Why a run of a thing for making is over.
MADE_NOT_WON = "it is for making things, and I have tried it out"


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
    #: The same screen's pieces of writing, each as it was read.
    ending_parts: list[str] = field(default_factory=list)
    #: Something went on going while she held a key: a world that waits for her and moves under her hand.
    under_her_hand: bool = False
    #: Screens this run reached that the game had not shown her before, told apart by their words.
    new_screens: int = 0
    #: Her bearings on the last screen she took in (core/cognition/her_bearings.py).
    bearings: Any = None
    #: How often play as it happens was held back for the way on a screen offered, by the screen's words.
    held_for_a_way_on: dict[str, int] = field(default_factory=dict)
    #: Since when its words have been known to set her to make something, not to win.
    for_making_since: float | None = None
    _clip: dict[str, float] | None = None
    _focused: bool = False
    _frames: Any = None
    _canvas: Any = None

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
        from core.perception.frames_as_they_are_drawn import CanvasFrames, PageFrames

        clip = await self._where()
        # The drawing read from its own canvas where it can be; else frames as
        # the browser draws them; a screenshot a picture only where neither.
        if self._canvas is None:
            self._canvas = CanvasFrames(self.page)
        if not self._canvas.unavailable:
            read = await self._canvas.look(clip)
            if read is not None:
                return read
        if self._frames is None:
            self._frames = PageFrames(self.page)
        if not self._frames.unavailable:
            streamed = await self._frames.look(clip)
            if streamed is not None:
                return streamed
        try:
            # At the page's own pixels: on a high-density screen a picture
            # at device pixels is four times the size for the same view, and
            # live play ran at thirteen pictures a second (2026-10-04).
            from core.perception.a_picture_of_her_page import picture_of

            picture = await picture_of(self.page, clip, css=True, kind="jpeg")
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            return None
        at = time.monotonic()
        if picture is None:
            return None
        return picture, at

    async def close(self) -> None:
        """Stop streaming frames, when the run is over."""
        if self._canvas is not None:
            await self._canvas.close()
            self._canvas = None
        if self._frames is not None:
            await self._frames.close()
            self._frames = None

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

    async def pressed(self, key: str) -> None:
        """A key pressed for the pursuit the way a person presses one to see what it does: held a moment, watched."""
        from core.agency.playing_as_it_happens import it_goes_while_held

        if await it_goes_while_held(self.look, self, key) and not self.under_her_hand:
            self.under_her_hand = True
            logger.info("something went on going while %s was held: the world waits for her and moves under her hand", key)

    async def _at(self, x: float, y: float) -> tuple[float, float]:
        clip = await self._where()
        return clip["x"] + min(1.0, max(0.0, x)) * clip["width"], clip["y"] + min(1.0, max(0.0, y)) * clip["height"]

    async def click(self, x: float, y: float) -> None:
        await self.page.mouse.click(*await self._at(x, y))
        self._focused = True

    async def point(self, x: float, y: float) -> None:
        await self.page.mouse.move(*await self._at(x, y))

    async def press(self, x: float, y: float) -> None:
        """The button pressed at a place and kept down, for a pull or a hold (core/agency/playing_by_shots.py)."""
        await self.page.mouse.move(*await self._at(x, y))
        await self.page.mouse.down()
        self._focused = True

    async def release(self) -> None:
        await self.page.mouse.up()

    # -- the pursuit's hooks ---------------------------------------------------

    def read(self, observation: dict[str, Any]) -> None:
        """What a look of the pursuit showed: the screen's words, and its ways back at the start."""
        from core.skills.screen_pursuit_bearings import restart_controls

        said = " ".join(str(observation.get("text") or "").split())
        if said and (not self.words or self.words[-1] != said):
            self.words.append(said)
            del self.words[:-12]
            self._a_screen_not_seen_before(said)
        if self.first_ways_back is None and observation.get("ok", True):
            self.first_ways_back = restart_controls(observation)

    def _a_menu_first(self) -> bool:
        """Whether to read the screen, and go on from it, before playing what moves on it.

        A screen not yet read is read first; a screen whose words offer a way on (Start, Play, Next) is a menu,
        whatever moves on it, and is clicked through first: LIVE 2026-10-08 a title's fireflies were played for
        fifty-three seconds with START under them. Twice at most for one screen, in case its way on does nothing.
        """
        from core.language.a_way_on import offers_a_way_on

        if not self.words:
            return True
        said = self.words[-1]
        if not offers_a_way_on(said):
            return False
        key = said[:80]
        self.held_for_a_way_on[key] = self.held_for_a_way_on.get(key, 0) + 1
        return self.held_for_a_way_on[key] <= 2

    def _a_screen_not_seen_before(self, said: str) -> None:
        """Count a screen the game has not shown her before: getting to one is getting somewhere, scored or not.

        LIVE 2026-10-08 a game's menus, story and player choice took most of a round, nothing was scored on
        them, and the game was left as its play began. Kept across the rounds of one game; two screens are one
        where they share most of their words (core/agency/where_things_lead.py).
        """
        from core.agency.where_things_lead import SAME_SCREEN, screen_words

        words = screen_words([], said)
        if len(words) < 2:
            return
        seen: list[frozenset[str]] = self.keep.setdefault("screens_seen", [])
        if any(len(words & before) >= SAME_SCREEN * len(words | before) for before in seen):
            return
        seen.append(words)
        del seen[:-60]
        self.new_screens += 1

    def run_is_over(self, observation: dict[str, Any]) -> bool:
        """Whether she has played, and the screen now offers a way to start again it did not offer at first."""
        if self.over_because:
            return True
        from core.language.how_a_game_ended import what_it_asks_of_a_player

        # A thing for making has no end of its own: a person tries it out a while and is done with it.
        if self.for_making_since is None and what_it_asks_of_a_player(" ".join(self.words)) == "make":
            self.for_making_since = time.monotonic()
        if self.for_making_since is not None and time.monotonic() - self.for_making_since >= TRYING_OUT_S:
            self.over_because = MADE_NOT_WON
            return True
        if not any(stretch.get("pictures") for stretch in self.stretches):
            return False
        from core.skills.screen_pursuit_bearings import restart_controls

        # Against what was on screen while she played, where she read it during
        # play: the screen that ended the last run already offered a way back,
        # so the first reading of this one is no measure of what is new.
        during = {said for stretch in self.stretches for said in stretch.get("words_seen") or ()}
        before = during if during else (self.first_ways_back or frozenset())
        offered = restart_controls(observation)
        appeared = {control for control in offered if control not in before}
        if offered and not appeared:
            logger.info("a way to start again is on screen (%s), but it was there while she played", ", ".join(sorted(offered)))
        if appeared:
            self.over_because = f"a way to start again appeared ({', '.join(sorted(appeared))})"
            self.ending_words = " ".join(str(observation.get("text") or "").split())
            self.ending_parts = _lines_of(observation.get("layout") or [])
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
        if not self.under_her_hand and self._a_menu_first():
            return
        # Played as it happens where it moves on its own, and where it moves for as long as she holds a key.
        if not self.under_her_hand and not await the_world_moves_on_its_own(self.look):
            # A world that waits until she sends something into it, where its words say so, is played by shots.
            await self._by_shots(now)
            return
        if not self.keep.get("hers") and not self.recalled:
            self.recalled = True
            self.keep.update(_what_she_kept_of(self.page))
        # What the place's screens said, and what taking stock found out (core/cognition/taking_stock.py), read alike.
        read = " ".join([self.goal, *self.words[-6:], str(self.keep.get("counsel") or "")])
        named, pointer_first = controls_named_in(read,
                                                keys_without_words=(), during_play=True)
        # The game's controls are every key any of its screens has named, not
        # only this screen's: LIVE 2026-10-04 a run begun from the end screen
        # ("Press SPACE to play again") was played with space alone, and the
        # arrows the title screen had named were never pressed.
        keys = list(dict.fromkeys([*(self.keep.get("named_keys") or []), *named]))
        if not keys:
            keys = controls_named_in("")[0]
        # Keys named only for doing something ("space to jump") say nothing of moving, and a person takes the arrows to
        # move: LIVE 2026-10-07 a platformer whose screen named space and up was played with those alone, its hero never
        # walked. Where nothing named moves her (no arrows, no WASD, no pointer), the arrows are tried too.
        if not pointer_first and not {"up", "down", "left", "right", "w", "a", "s", "d"} & set(keys):
            keys = [*keys, "left", "right", "up", "down"]
        pointer_first = pointer_first or bool(self.keep.get("pointer_named"))
        self.keep["named_keys"], self.keep["pointer_named"] = keys, pointer_first
        logger.info("it moves on its own: playing it as it happens with %s%s", keys, " and the pointer" if pointer_first else "")
        stretch = await play_as_it_happens(
            self.look, self, keys=keys, seconds=min(STRETCH_S, self.ends_at - now),
            say=_said_while_playing, read_words=recognize_text, keep=self.keep,
            pointer_first=pointer_first, getting_somewhere=_getting_somewhere,
            told=read,
            waits_for_her=self.under_her_hand,
        )
        self.stretches.append(stretch)
        if (stretch.get("runtime_checks") or {}).get("violations"):
            self.over_because = "runtime contract violated"
        _keep_what_she_learned(self.page, self.keep)
        logger.info("a stretch played as it happened: %s", {k: stretch.get(k) for k in ("seconds", "ended", "hers", "keys_that_move_her", "pictures_a_second", "learned", "gains", "losses")})
        if not stretch.get("hers") and not stretch.get("gains") and not stretch.get("losses"):
            self.quiet_until = time.monotonic() + LEAVE_A_MOVING_MENU_S
            self.under_her_hand = False

    async def _by_shots(self, now: float) -> None:
        """Shots, where the place's words speak of sending a thing by a press pulled or held and let go."""
        from core.agency.playing_by_shots import play_by_shots, sends_by_letting_go
        from core.perception.what_the_pixels_show import recognize_text

        told = " ".join([*self.words[-6:], str(self.keep.get("counsel") or "")])
        if not sends_by_letting_go(told) or now < self.quiet_until:
            return
        logger.info("it waits for her to send something: playing it by shots")
        shots = self.keep.setdefault("by_shots", {})
        stretch = await play_by_shots(self.look, self, seconds=min(STRETCH_S, self.ends_at - now), keep=shots,
                                      say=_said_while_playing, read_words=recognize_text)
        logger.info("shots played: %s", stretch)
        self.stretches.append({"pictures": stretch.get("shots", 0), "gains": stretch.get("gains", 0), "losses": 0,
                               "seconds": time.monotonic() - now, "ended": stretch.get("ended"), "by_shots": stretch})
        _keep_what_she_learned(self.page, self.keep)
        if not stretch.get("sends_from"):
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
        if last.get("standing"):
            parts.append(str(last["standing"]).rstrip("."))
        elif counters:
            parts.append(", ".join(f"{name} {value}" for name, value in counters.items() if not name.startswith("number")))
        if self.over_because:
            parts.append(f"the run is over: {self.over_because}")
        return "; ".join(part for part in parts if part)


def _lines_of(regions: list[dict[str, Any]]) -> list[str]:
    """A screen's writing as lines: pieces at one height, left to right.

    Text recognition hands back pieces, and a title in large letters comes
    back as one piece a word ("You", "win!") read in among the smaller line
    below it. Pieces whose middles are within half a piece's height of each
    other are one line.
    """
    pieces = []
    for region in regions:
        text = str(region.get("text") or "").strip()
        if not text:
            continue
        height = float(region.get("height") or 0.0)
        middle = float(region.get("center_y") if region.get("center_y") is not None else float(region.get("y") or 0.0) + height / 2)
        left = float(region.get("x") or region.get("center_x") or 0.0)
        pieces.append((middle, height, left, text))
    lines: list[list[tuple[float, float, float, str]]] = []
    for piece in sorted(pieces):
        for line in lines:
            if abs(line[0][0] - piece[0]) <= 0.5 * max(line[0][1], piece[1], 1e-6):
                line.append(piece)
                break
        else:
            lines.append([piece])
    return [" ".join(text for _m, _h, _l, text in sorted(line, key=lambda p: p[2])) for line in lines]


def _this_game(page: Any) -> str:
    """The name her memory keeps this game under: its page, and for a file, that file as it now is.

    A file changed is another game. LIVE 2026-10-06 a Pong she had just
    mended was played from what she had kept of it broken ("the white bars
    cost me"), because both were kept under one address.
    """
    url = str(getattr(page, "url", "") or "")
    if url.startswith("file://"):
        import hashlib
        from pathlib import Path
        from urllib.parse import unquote, urlparse

        try:
            held = Path(unquote(urlparse(url).path)).read_bytes()
            return f"played as it happens at {url} ({hashlib.sha256(held).hexdigest()[:12]})"
        except OSError:
            pass
    return f"played as it happens at {url}"


def _what_she_kept_of(page: Any) -> dict[str, Any]:
    from core.agency.what_she_keeps_of_a_game import INDEXED_TABLES, kept_from
    from core.runtime.what_she_learned import recall

    held = recall(_this_game(page), indexed_tables=INDEXED_TABLES)
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


async def looked_at_as_it_happens(look: Any) -> dict[str, Any]:
    """``await look()``, with the hand-over's reflexes around it when there are any.

    What moves on its own is played first, and the look is of the screen that
    play ends on; the reflexes then read what the look saw.
    """
    reflexes = AS_IT_HAPPENS.get()
    if reflexes is not None:
        await reflexes.while_it_moves()
    seen = await look()
    if reflexes is not None:
        reflexes.read(seen)
    return seen


def a_handed_over_run_is_over(observation: dict[str, Any]) -> bool:
    """A run played and over ends a hand-over; what follows belongs to whoever handed it over."""
    reflexes = AS_IT_HAPPENS.get()
    return reflexes is not None and reflexes.run_is_over(observation)
