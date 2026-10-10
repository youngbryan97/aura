"""Taking a thing and using it on another: two clicks, the first on what was taken, the second on what it is used on.

A game of finding and using things (a key for a door, glasses for a robot, a can
of paint for a wall) and a program that works the same way (pick a tool, then
apply it; choose a file, then the folder) share one act: take, then use. Taken,
a thing goes from where it was and turns up elsewhere on the screen (a bar along
an edge, a hand, a list), or stays where it is and is marked as chosen.

So after a click on a thing she looks at what came of it. Gone from where it was,
with one new thing to click in its place elsewhere, the new thing is it, taken.
Still there, on the same screen with the same labels, but the screen changed:
it is chosen. Either way it is hers to use, and every other thing on the screen
is somewhere to use it: 'use "glasses" on "robot"', a click on the one and then
on the other. Which use to make is a choice among those on offer, made as any
other move is.

Nothing here knows what a key, a door or a tool is.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = ["TakingAndUsing", "a_use_of", "what_is_used"]

#: Things taken, the latest kept.
TAKEN_KEPT = 6
#: The most uses offered at once: the latest things taken, each on the things on the screen.
USES_OFFERED = 12

_USE = re.compile(r'^use "(.+)" on "(.+)"$')

#: Words that say a place's things are taken and used on others: an inventory, objects found and used, a tool picked.
_SPEAKS_OF_USING = re.compile(
    r"\b(?:inventory|use (?:the |an? |your )?(?:objects?|items?|things?|tools?)|objects you find|items? you find|"
    r"pick (?:it |them |things |objects |items )?up|place them in your|select (?:an? |the )?(?:tool|item|weapon)|"
    r"equip|combine|use (?:it|them) (?:on|to|with))\b", re.I)


def a_use_of(item: str, target: str) -> str:
    """The move that uses one thing on another, named by what is written on each."""
    return f'use "{" ".join(str(item).split())}" on "{" ".join(str(target).split())}"'


def what_is_used(move: str) -> tuple[str, str] | None:
    """What a use move uses and on what, or None when the move is not a use."""
    found = _USE.match(str(move or "").strip())
    return (found.group(1), found.group(2)) if found else None


@dataclass
class TakingAndUsing:
    """What she has taken by clicking it, under what it is called now, for WhatWorksHere."""

    #: What a thing was called when she clicked it -> what it is called now.
    taken: dict[str, str] = field(default_factory=dict)
    clicked_last: str = ""
    labels_at_click: tuple[str, ...] = ()
    click_changed: bool = False
    #: Whether the place's words, or what taking stock found, speak of using things on others: then a thing clicked
    #: and still where it was may have been chosen. Elsewhere such a click was only clicked.
    using_said: bool = False

    def told_of_using(self, words: str) -> None:
        """Words read at this place: once they speak of using things, it is a place where things are chosen and used."""
        self.using_said = self.using_said or bool(_SPEAKS_OF_USING.search(words or ""))

    def clicked_on(self, move: str, labels: Sequence[str], changed: bool) -> None:
        """A click made with ``labels`` on the screen, and whether the screen answered it."""
        self.clicked_last, self.labels_at_click, self.click_changed = move, tuple(labels), changed

    def noticed_taking(self, now: Sequence[str]) -> None:
        """At the look after a click: whether what was clicked was taken, and what it is called now."""
        from core.agency.what_i_can_do_here import what_is_clicked

        clicked, before = self.clicked_last, set(self.labels_at_click)
        self.clicked_last = ""
        from core.agency.what_i_can_do_here import _not_read_only

        label = what_is_clicked(clicked) if clicked else None
        # Writing there to be read is not a thing taken, whatever changed as it was clicked: LIVE 2026-10-10 "LEVEL" was
        # taken off a game's bar and used on "HITS LEFT".
        if label is None or not self.click_changed or not _not_read_only(clicked):
            return
        came = [move for move in now if move not in before]
        if clicked not in now and len(came) == 1:
            held = what_is_clicked(came[0])
        elif clicked in now and set(now) == before and self.using_said:
            # Chosen where it is, where the place speaks of using things: LIVE 2026-10-09 on a game's end screen every
            # click that set something moving "took" a shape, and she used it on the word SORRY.
            held = label
        else:
            return
        if held:
            self.taken.pop(label, None)
            self.taken[label] = held
            while len(self.taken) > TAKEN_KEPT:
                self.taken.pop(next(iter(self.taken)))

    def uses(self, on_screen: Sequence[str]) -> tuple[str, ...]:
        """Each thing taken, still on the screen, used on each other thing there worth clicking; the latest taken first."""
        from core.agency.what_i_can_do_here import _not_read_only, a_click_on, what_is_clicked

        offered: list[str] = []
        for held in reversed(list(self.taken.values())):
            if a_click_on(held) not in on_screen or not _not_read_only(a_click_on(held)):
                continue
            for move in on_screen:
                target = what_is_clicked(move)
                if target and target != held and _not_read_only(move):
                    offered.append(a_use_of(held, target))
        if "serving" in getattr(self, "mechanics_said", set()):
            from core.agency.putting_things_in_place import served_first

            offered = served_first(offered, getattr(self, "looked", {}), what_is_used)
        return tuple(offered[:USES_OFFERED])
