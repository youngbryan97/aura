"""Which part of the screen answers to her.

A reading of a screen is everything on it. On the page holding a game that is
the board, the score, two advertising rails, a cookie footer and a copyright
line, and the situation she reasons about is all of it — so what she recalls
about "a board like this one" is dominated by whichever advertisement was
loaded at the time, and two readings of the same board look like different
situations because the advertising rotated.

Nothing about the page says which part is the task. But something about HER
does: the part that changes when she acts is the part she is acting on. Ads
change too, and pages animate, so a single coincidence proves nothing — it
takes repetition, and the answer is the region that keeps moving right after
she does something and does not otherwise.

That is general to any screen. It finds the board in a game, the editor in a
window full of panels, the results in a page of chrome. It knows nothing
about any of them; it knows what she did and what happened next.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from core.perception.what_is_there import Arrangement, arranged

logger = logging.getLogger("Aura.Responds")

#: How many acts it takes before the answer is worth using. One coincidence
#: is a coincidence.
ENOUGH_ACTS = 4

#: How much of the changing area to keep. Padding a tight band by a little
#: allows for a value appearing one row further out than any seen so far.
MARGIN = 0.03

#: How responsive a place has to be, against the most responsive place on
#: the screen, before it counts as answering to her.
#:
#: Everything on a page changes eventually — advertising rotates, clocks
#: tick, banners cycle — so "it changed" is not the test. The test is whether
#: it changes when SHE does something, and how often is only meaningful
#: beside the rest of the screen. A fixed share of her acts cannot serve
#: both: a game board where each square changes a third of the time and an
#: editor where one line changes every keystroke are the same question with
#: different numbers. Measured against the busiest place, both answer it.
ANSWERS_OFTEN = 0.34


@dataclass
class Responsive:
    """Where her actions have been having their effects."""

    acts: int = 0
    #: How often the text at each position changed on a move that had an
    #: effect.
    answered: dict[tuple[int, int], int] = field(default_factory=dict)
    #: How often it changed on a move that had NO effect. Those moves are the
    #: control: whatever still changed then was changing on its own.
    regardless: dict[tuple[int, int], int] = field(default_factory=dict)
    #: How many of each kind of move there have been, so the two rates can be
    #: compared rather than the two counts.
    effective: int = 0
    idle: int = 0

    #: How many acts in a row can have no effect before the world she is
    #: working in has stopped answering her altogether.
    #:
    #: One is a bad move. Two is a bad idea. Several in a row, with nothing
    #: anywhere on the screen responding, is not a run of bad moves — it is a
    #: thing that has ended. A finished game, a session that expired, a form
    #: already submitted, a connection that dropped.
    DEAD_AFTER = 4

    #: Acts since anything last answered her.
    unanswered: int = 0
    #: Every act she has been seen to take here, and the ones in the run of
    #: acts that changed nothing. "Nothing I do changes anything" is a claim
    #: about the things she does, so it needs the things she does.
    tried: set[str] = field(default_factory=set)
    unanswered_by: set[str] = field(default_factory=set)

    #: The places that answer to her, as of the last time the band was worked
    #: out. Kept because a BOX is not the same as a set of places: furniture
    #: sitting inside the thing's outline is inside the box, and a lattice
    #: built from it is not the thing's own.
    _places: frozenset[tuple[int, int]] = field(default_factory=frozenset)

    def as_memory(self) -> dict[str, Any]:
        """Where she found things happen, in a form that survives the process."""
        return {
            "acts": self.acts,
            "effective": self.effective,
            "idle": self.idle,
            "answered": {f"{x},{y}": n for (x, y), n in self.answered.items()},
            "regardless": {f"{x},{y}": n for (x, y), n in self.regardless.items()},
        }

    @classmethod
    def from_memory(cls, held: dict[str, Any], trust: float = 1.0) -> Responsive:
        """Where she found things happen last time, discounted.

        A page can be rebuilt between visits, so what she knew about where it
        answers is a starting point rather than a finding. Discounted, a few
        acts that disagree move the band.
        """
        if not isinstance(held, dict):
            return cls()
        share = max(0.0, min(1.0, float(trust)))

        def places(counts: Any) -> dict[tuple[int, int], int]:
            found: dict[tuple[int, int], int] = {}
            if not isinstance(counts, dict):
                return found
            for where, value in counts.items():
                try:
                    x, y = (int(part) for part in str(where).split(","))
                except (TypeError, ValueError):
                    continue
                if isinstance(value, (int, float)):
                    found[(x, y)] = int(round(float(value) * share))
            return found

        return cls(
            acts=int(round(float(held.get("acts") or 0) * share)),
            answered=places(held.get("answered")),
            regardless=places(held.get("regardless")),
            effective=int(round(float(held.get("effective") or 0) * share)),
            idle=int(round(float(held.get("idle") or 0) * share)),
        )

    def began_again(self) -> None:
        """A new world, in the same place as the one that ended.

        What she learned about WHERE things happen still holds — the board is
        where the board was. What does not carry over is the verdict that
        nothing answers, and while it stood she was never offered a move
        again: only the ways out, forever. LIVE 2026-08-26: she noticed a
        finished game, chose to start a new one, clicked it, and then made no
        move at all for the rest of the run.
        """
        self.unanswered = 0
        self.unanswered_by = set()

    def settled(self) -> bool:
        return self.effective >= ENOUGH_ACTS and bool(self.answered)

    def nothing_answers(self) -> bool:
        """Whether the thing she is working in has stopped responding at all.

        Read from what happened rather than from what the screen says. A page
        that has ended says so in its own words — "Game Over", "Session
        expired", "Thanks for your submission" — and there is no list of
        those words that covers the next one. What every ending has in common
        is that nothing she does changes anything any more.
        """
        if self.unanswered < self.DEAD_AFTER:
            return False
        if not self.tried:
            # Nobody said which act was taken, so the count is all there is.
            return True
        # Every act she has, and every one of them did nothing.
        #
        # Without this, a run of one repeated act reads as the world having
        # ended. LIVE 2026-09-04 on a live board: four presses of up, the
        # board correctly refusing all four because there was nothing above
        # anything, "nothing I do is changing anything here — this attempt is
        # over", and a New Game clicked over a game that was very much alive.
        # Four acts that did nothing is a fact about those acts until the
        # acts are all of them.
        return self.unanswered_by >= self.tried

    def band(self) -> tuple[float, float, float, float] | None:
        """The area that answers to her, as (left, top, right, bottom).

        Only the positions that answer to a good share of her acts. Everything
        on a page changes eventually, so "it changed" would widen the band to
        the whole screen given long enough.
        """
        if not self.settled():
            return None
        # How much more often a place moves when she has an effect than when
        # she does not. A place that moves either way is moving on its own.
        idle_rate = {
            where: times / float(max(1, self.idle)) for where, times in self.regardless.items()
        }
        because_of_her = {
            where: (times / float(max(1, self.effective))) - idle_rate.get(where, 0.0)
            for where, times in self.answered.items()
        }
        because_of_her = {where: rate for where, rate in because_of_her.items() if rate > 0.0}
        if not because_of_her:
            return None
        busiest = max(because_of_her.values())
        repeated = [
            where for where, rate in because_of_her.items() if rate >= busiest * ANSWERS_OFTEN
        ]
        if not repeated:
            return None
        self._places = frozenset(repeated)
        xs = _middle(sorted(where[0] / 100.0 for where in repeated))
        ys = _middle(sorted(where[1] / 100.0 for where in repeated))
        return (
            max(0.0, xs[0] - MARGIN),
            max(0.0, ys[0] - MARGIN),
            min(1.0, xs[1] + MARGIN),
            min(1.0, ys[1] + MARGIN),
        )


#: What share of the responsive positions to keep at each edge.
#:
#: A bounding box around everything that answered is as wide as its worst
#: outlier, and a screen read by OCR always has one: a run of text jitters by
#: a pixel, a banner redraws, a word is read differently between frames.
#: Measured on a real page, eight moves of that noise stretched the answer
#: from the board to the whole window. Dropping the outermost tenth at each
#: edge keeps the part that answers and loses the strays.
TRIM = 0.1


def the_places_that_answer(state: Any) -> frozenset[tuple[int, int]]:
    """Where things happen because of her, one place at a time.

    The band is these places' outline, and an outline is coarser than they
    are: a score, a title and a New Game button all sit inside a board's
    outline, and a grid worked out from them is not the board's grid. LIVE
    2026-08-31 on a native app, with a clean reading of one window: the tiles
    landed on columns 0, 3, 4 and 6 of a nine-column lattice the furniture
    had defined, and no rule about sliding along a row could match, because
    the rows were not rows of the board.
    """
    band = getattr(state, "band", None)
    if callable(band):
        band()  # works the places out, as a side effect of settling the box
    return frozenset(getattr(state, "_places", frozenset()) or frozenset())


def _middle(values: Sequence[float]) -> tuple[float, float]:
    """The range these values occupy, ignoring the strays at each end."""
    if not values:
        return (0.0, 1.0)
    if len(values) < 5:
        return (values[0], values[-1])
    cut = max(1, int(len(values) * TRIM))
    kept = values[cut : len(values) - cut] or values
    return (kept[0], kept[-1])


def places_and_text(observation: dict[str, Any]) -> dict[tuple[int, int], str]:
    """The reading as text by position, rounded so a pixel of drift is not a change.

    Public because anything comparing one reading against the next has to use
    the same places these do. Two callers rounding differently would produce
    two sets that cannot be intersected, and the mistake would look like a
    board with nothing in it rather than like a bug.
    """
    found: dict[tuple[int, int], str] = {}
    for region in observation.get("layout") or []:
        text = str(region.get("text") or "").strip()
        if not text:
            continue
        try:
            x = float(region.get("center_x", region.get("x", 0.0)))
            y = float(region.get("center_y", region.get("y", 0.0)))
        except (TypeError, ValueError):
            continue
        found[(round(x * 100), round(y * 100))] = text
    return found


def noticed(
    state: Responsive,
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    worked: bool = True,
    acting: str = "",
) -> Responsive:
    """Record what changed between one reading and the next, after she acted.

    ``worked`` says whether that act had any effect. A move that changed
    nothing is the control this needs: everything that still changed across
    it was changing on its own, and a page whose advertising animates as
    often as the task does cannot be separated any other way.

    ``acting`` names the act, so a run of nothing happening can be told apart
    from a run of ONE THING doing nothing. A caller that does not say leaves
    the verdict on the count alone, which is what it always was.
    """
    was, now = places_and_text(before), places_and_text(after)
    if not was and not now:
        return state
    state.acts += 1
    # Something answered only if what changed is not the sort of thing that
    # changes anyway.
    #
    # A reading of a screen holds a clock, a tab strip, a rotating banner. On
    # a board that had finished, one of those ticked over between every pair
    # of readings, so "the screen changed" was true after every act and she
    # went on playing a game that was over. A place that has changed on every
    # comparison so far is not evidence that this act did anything.
    answered_now = False
    for where in set(was) | set(now):
        if was.get(where) == now.get(where):
            continue
        # Against the acts that had NO effect, which is the only control
        # there is. A place that also changes when she does nothing is
        # changing on its own; one that has never been seen to do that is
        # answering her.
        #
        # This used to ask whether the place had changed on every comparison
        # so far, effect or none, and that is the wrong question on anything
        # she is actually driving: a board answers in the same sixteen
        # places every single move, so by the fourth move every place that
        # mattered had been disqualified and the run declared the world
        # dead. The clock in a corner that ticks over regardless is what the
        # test is for, and the idle acts are where it shows itself.
        #
        # ``max(1, ...)`` is what makes a first act possible. With no idle
        # comparison to hold anything against, nothing has been shown to
        # change on its own, so a change is evidence — where the old
        # arithmetic made act one unanswerable by construction.
        idle_seen = state.regardless.get(where, 0)
        if idle_seen < max(1, state.idle):
            answered_now = True
            break
    if acting:
        state.tried.add(acting)
    if answered_now:
        state.unanswered = 0
        state.unanswered_by.clear()
    else:
        state.unanswered += 1
        if acting:
            state.unanswered_by.add(acting)
    if worked:
        state.effective += 1
    else:
        state.idle += 1
    counted = state.answered if worked else state.regardless
    for where in set(was) | set(now):
        if was.get(where) != now.get(where):
            counted[where] = counted.get(where, 0) + 1
    return state


def _changes_anyway(state: Responsive | None) -> set[tuple[int, int]]:
    """Positions that have changed on every comparison so far.

    A clock, a tab strip, a rotating banner. They are not evidence that
    anything she did had an effect, and leaving them in what she reads makes
    every move look like it worked — which is worse than useless, because it
    is the measurement everything else learns from.
    """
    if state is None or state.acts < 2:
        return set()
    seen = set(state.answered) | set(state.regardless)
    return {
        where
        for where in seen
        if state.answered.get(where, 0) + state.regardless.get(where, 0) >= state.acts - 1
    }


def within(
    observation: dict[str, Any],
    band: tuple[float, float, float, float] | None,
    state: Responsive | None = None,
) -> str:
    """The reading, kept to the part that answers to her.

    Before the answering part is known, everything is kept except what has
    been changing regardless of her — because a guess about where the task is
    would be worse than reading everything, and reading a clock as if it were
    the task is worse than either.
    """
    text = str(observation.get("text") or "")
    if band is None:
        background = _changes_anyway(state)
        if not background:
            return text
        kept: list[str] = []
        for region in observation.get("layout") or []:
            said = str(region.get("text") or "").strip()
            if not said:
                continue
            try:
                where = (
                    round(float(region.get("center_x", region.get("x", 0.0))) * 100),
                    round(float(region.get("center_y", region.get("y", 0.0))) * 100),
                )
            except (TypeError, ValueError):
                continue
            if where not in background:
                kept.append(said)
        return " ".join(kept) if kept else text
    left, top, right, bottom = band
    inside: list[tuple[float, float, str]] = []
    for region in observation.get("layout") or []:
        said = str(region.get("text") or "").strip()
        if not said:
            continue
        try:
            x = float(region.get("center_x", region.get("x", 0.0)))
            y = float(region.get("center_y", region.get("y", 0.0)))
        except (TypeError, ValueError):
            continue
        if left <= x <= right and top <= y <= bottom:
            inside.append((y, x, said))
    return _laid_out(inside) or text


def _where_the_lines_put_it(text: Any) -> list[dict[str, Any]]:
    """Places for a reading that came back as prose.

    A line is a row and the nth word of a line is its nth column. By its place
    in the line rather than by where its characters sit: text is not drawn at
    an even pitch, so a row reading "16 32 64 4" and a row reading "2 4 8 2"
    put their four values at quite different distances along, and binning
    those by distance makes five columns out of four. Counting instead is
    exact for anything laid out and no worse than honest for anything else.

    The positions are shares of the reading rather than of the window, because
    that is the frame the text itself is in, and every later step works in
    shares.
    """
    lines = [line.split() for line in str(text or "").splitlines()]
    lines = [words for words in lines if words]
    if not lines:
        return []
    widest = max(len(words) for words in lines)
    tall = len(lines)
    return [
        {
            "text": said,
            "center_x": (column + 0.5) / widest,
            "center_y": (row + 0.5) / tall,
        }
        for row, words in enumerate(lines)
        for column, said in enumerate(words)
    ]


def what_is_there(
    observation: dict[str, Any],
    band: tuple[float, float, float, float] | None,
    like: Arrangement | None = None,
    answering: frozenset[tuple[int, int]] | None = None,
    lattice: Any = None,
) -> Arrangement:
    """The same reading as :func:`within`, with a place for each thing in it.

    :func:`within` hands back the string she reads. This hands back what it
    was made from, so a plan about a corner or a bottom row can be checked
    against the thing itself rather than against prose about it.

    ``lattice`` is the grid she is holding, and where she has one this reading
    is placed into it and nothing else happens. That is the difference between
    a video and a pile of photographs: the squares are there when they are
    empty, so a board whose top row is empty this turn still has four rows and
    a tile that has not moved keeps its address. Worked out afresh each time,
    a reading is internally consistent and incomparable with the one before
    it, and no rule about movement can be checked against a frame of reference
    that moves too.

    ``answering`` is the set of places that move because of her, and where it
    is known the grid is worked out from those alone. This matters more than
    it sounds: the band is their OUTLINE, and a score, a title and a New Game
    button all sit inside a board's outline. LIVE 2026-08-31 on a clean
    reading of one window, the tiles landed on columns 0, 3, 4 and 6 of a
    nine-column lattice the furniture had defined, and no rule about sliding
    along a row could match, because those were not the board's rows.

    Before she has settled which places answer, everything inside the band is
    used, which is what she has.
    """
    inside: list[tuple[float, float, str]] = []
    positioned = observation.get("layout") or []
    if not positioned:
        # A reading with words in it and no coordinates is still a reading.
        #
        # Everything downstream — whether anything moved, what the rule of
        # this world is, what a move would lead to, how near the goal she is
        # — is asked of the arrangement and of nothing else. So a reader that
        # hands back the text without saying where any of it sat left her
        # entirely blind while the words were sitting right there, and every
        # move she made came back "nothing changed" because two empty
        # arrangements are equal. Lines and the gaps between them are the
        # layout the text itself carries; used, it is coarse, and coarse is
        # the difference between a world she can model and no world at all.
        positioned = _where_the_lines_put_it(observation.get("text"))
        # And the band does not apply to them. A band names part of a WINDOW;
        # these places are shares of the reading, which is whatever was
        # captured — so there is no part of it left to select, and selecting
        # anyway crops the whole reading away. Measured: a band of the middle
        # third discarded every invented place and put her back at nothing.
        band = None
    for region in positioned:
        said = str(region.get("text") or "").strip()
        if not said:
            continue
        try:
            x = float(region.get("center_x", region.get("x", 0.0)))
            y = float(region.get("center_y", region.get("y", 0.0)))
        except (TypeError, ValueError):
            continue
        if band is None:
            inside.append((y, x, said))
            continue
        left, top, right, bottom = band
        if left <= x <= right and top <= y <= bottom:
            inside.append((y, x, said))
    # The part of it that is laid out, out of the page around it.
    #
    # A thing she acts in sits at an even pitch across and down and is full of
    # its own places; a page is not. Read whole, 2048game.com came back as
    # forty-four columns by thirty-seven rows, so of thirty moves only five
    # could be compared with each other and the rule that governs the board
    # scored nought out of five. She was playing correctly and learning
    # nothing from it.
    _note_what_was_seen(inside, band)
    if lattice is not None and getattr(lattice, "held", False):
        placed = lattice.fit(inside)
        if placed is not None:
            return placed
    if answering:
        # A place is where a thing sits, to the nearest hundredth of the
        # window — the same quantisation the answering places are counted in.
        only = [
            (y, x, said)
            for y, x, said in inside
            if (int(round(x * 100)), int(round(y * 100))) in answering
        ]
        # Cropping to nothing is not a reading.
        if len(only) >= 4:
            return arranged(only, like=like)
    return arranged(inside, like=like)


def what_the_page_is_showing(
    said: Sequence[tuple[float, float, str]],
    band: tuple[float, float, float, float] | None,
    like: Arrangement | None = None,
    lattice: Any = None,
) -> Arrangement:
    """The same arrangement, built from what the page reported rather than seen.

    The page reader already hands back ``(y, x, text)`` in the shape the screen
    reader produces, which is what makes this three lines: everything
    downstream is unchanged and does not know which instrument was used.

    The held grid is used here for the same reason it is used there, and it
    matters more: two readers that each work out their own grid put the same
    board in two frames of reference, and which one she gets depends on which
    reader happened to see more this turn. Placed into the one she is holding,
    they agree by construction.
    """
    inside: list[tuple[float, float, str]] = []
    for entry in said or ():
        try:
            y, x, text = float(entry[0]), float(entry[1]), str(entry[2]).strip()
        except (IndexError, TypeError, ValueError):
            continue
        if not text:
            continue
        if band is None:
            inside.append((y, x, text))
            continue
        left, top, right, bottom = band
        if left <= x <= right and top <= y <= bottom:
            inside.append((y, x, text))
    # The part of it that is laid out, out of the page around it.
    #
    # A thing she acts in sits at an even pitch across and down and is full of
    # its own places; a page is not. Read whole, 2048game.com came back as
    # forty-four columns by thirty-seven rows, so of thirty moves only five
    # could be compared with each other and the rule that governs the board
    # scored nought out of five. She was playing correctly and learning
    # nothing from it.
    _note_what_was_seen(inside, band)
    if lattice is not None and getattr(lattice, "held", False):
        placed = lattice.fit(inside)
        if placed is not None:
            return placed
    return arranged(inside, like=like)


def _laid_out(cells: Sequence[tuple[float, float, str]]) -> str:
    """The part that answers to her, arranged the way it is arranged.

    One rendering of :func:`core.perception.what_is_there.arranged`, kept here
    because a string is what reaches her prompt. The structure behind it is
    what everything else should be reading: a corner is a place code can ask
    about, and an approach phrased about one can be checked.
    """
    return arranged(cells).as_text()



def describe(band: tuple[float, float, float, float] | None) -> str:
    if band is None:
        return ""
    left, top, right, bottom = band
    return f"the part of the screen that responds to me ({left:.2f}–{right:.2f} across, {top:.2f}–{bottom:.2f} down)"


def _note_what_was_seen(
    cells: Sequence[tuple[float, float, str]],
    band: tuple[float, float, float, float] | None = None,
) -> None:
    """Write down one reading, when asked to, so a crop can be fixed against
    what she actually sees rather than against what a browser reports.

    Off unless AURA_NOTE_READINGS names a file. Her screen access is hers, so
    a reading cannot be sampled from beside her.
    """
    import os

    where = os.environ.get("AURA_NOTE_READINGS")
    if not where:
        return
    try:
        from core.runtime.file_write_gateway import get_file_write_gateway

        get_file_write_gateway().append_text(
            where,
            json.dumps(
                {
                    "band": list(band) if band else None,
                    "cells": [[round(y, 6), round(x, 6), t] for y, x, t in cells],
                }
            )
            + "\n",
            source="where_it_responds",
        )
    except (OSError, ImportError, RuntimeError, ValueError):
        # not a failure: writing readings down is a favour to whoever is
        # debugging, and it must never affect what she sees.
        pass
