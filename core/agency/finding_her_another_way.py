"""Finding which thing is hers where her trials of her keys could not: by its kind, and by what her presses do to it.

Her trials compare a thing's speeds under each key, press by press, and a thing
whose number is split by a meeting (a block passing through a runner) has its
presses split between two numbers and is found by neither. Two other readings of
the same evidence find it: the kind whose presses answer her, where one thing of
that kind is on the screen; and the one thing of a kind that every press of a key
moves alike (core/agency/what_a_press_does.py).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from core.agency.causal_identification import CausalWitness

logger = logging.getLogger("core.agency.which_one_answers_to_her")

__all__ = ["FoundAnotherWay"]


#: How long after a life lost she may be set back somewhere else, in seconds: a game redraws her within moments.
SET_BACK_S = 3.0


class FoundAnotherWay:
    """Ways of finding which thing is hers beside her trials, for WhichIsHers."""

    #: When she last lost a life, as her counters said.
    set_back_at: float = float("-inf")

    def lost_a_life(self, at: float) -> None:
        """Her counters say she lost: for a moment, she may be set back anywhere on the screen."""
        self.set_back_at = at

    def just_set_back(self, at: float | None) -> bool:
        """Whether a life was lost moments ago, so a lone lookalike far from where she was is her, set back."""
        return at is not None and 0.0 <= at - self.set_back_at <= SET_BACK_S

    def _the_one_of_a_kind_that_answers(self, moves: Any) -> int | None:
        """Where no one thing's presses answer her keys, the kind whose do, where one thing of it is on the screen.

        A thing's number does not outlast its being lost to sight: met by
        something, a body is one patch with it for a picture and comes out
        under a new number, and its presses are split between the two
        (offline 2026-10-09, a runner met by the first block during her trials
        of her keys: one press of each, under two numbers, and no answer). Its
        kind outlasts that, and a kind with one thing of it on the screen is
        that thing.
        """
        witnesses = []
        for kind, speeds in self._by_kind.items():
            things = [t for t in moves.things.values() if t.kind == kind and t.moved and t.number not in self.not_mine]
            if len(things) != 1:
                continue
            f, widest = speeds.ratio()
            trials: dict[str, int] = defaultdict(int)
            for (key, _press), values in speeds.by_press.items():
                if len(values) >= 2:
                    trials[key] += 1
            witnesses.append(CausalWitness(things[0].number, self.identification.epoch, tuple(sorted(trials.items())), f, widest))
        return self.identification.choose(witnesses, visible=set(moves.things), established=None, excluded=self.not_mine)

    def moved_by_her_presses(self, thing: Any, at: float) -> None:
        """``thing`` is the one thing of a kind her presses move the same way every time: it is hers.

        The same evidence as her trials, read another way: a press is a trial,
        and a thing a key moves alike at every press answers to her. Offline
        2026-10-09 a runner whose number was split by a block during her
        trials was never found by them, while every press of space was seen
        to lift its kind and let it down.
        """
        if self.number is not None or thing.number in self.not_mine:
            return
        self.number, self.kind = thing.number, thing.kind
        self.identification.receipts.append({"epoch": self.identification.epoch, "reason": "moved alike by every press",
                                             "selected": thing.number})
        logger.info("her thing is %s: her presses move its kind alike every time", thing.number)

    def seen_to_be_her(self, thing: Any, at: float, what: str = "") -> None:
        """``thing`` is what her eyes took for the player's own (core/perception/where_i_am_on_screen.py), and no trial
        of her keys has found anything that answers to her: it is taken for hers, until something answers otherwise.

        A person knows which one they are before pressing anything; LIVE 2026-10-10 a runner whose fruit jumped at a
        click, a lander that fell before her keys were tried and a car she found only in the second round were each
        played for minutes without knowing which one was her.
        """
        if self.number is not None or thing.number in self.not_mine:
            return
        self.number, self.kind = thing.number, thing.kind
        self.identification.receipts.append({"epoch": self.identification.epoch,
                                             "reason": f"her eyes took it for the player's own ({what})",
                                             "selected": thing.number})
        logger.info("her thing is %s: her eyes took it for the player's own (%s)", thing.number, what)

