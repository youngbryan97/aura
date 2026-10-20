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
from typing import Any

from core.agency.controls_measured_on_one_thing import own_control_witness, own_trajectory_witness
from core.perception.translation_from_visible_support import motion_was_measured
from core.perception.what_moves_in_the_picture import TOLD_FROM_STANDING_PX
from core.verify import invariant

logger = logging.getLogger("core.agency.which_one_answers_to_her")

__all__ = ["FoundAnotherWay"]


#: How long after a life lost she may be set back somewhere else, in seconds: a game redraws her within moments.
SET_BACK_S = 3.0


def found_by_press_curves(run: Any, moves: Any, hers: Any, at: float, *, old_for: float) -> None:
    """An entity's own curves suggest a trial before pooled kind histories."""
    if hers.number is not None or hers.follows_pointer or not run.presses.curves:
        return
    kinds = run.presses.kinds_it_moves(TOLD_FROM_STANDING_PX * 2)
    # What an input brings out was not there before it. Offline 2026-10-11
    # repeated emissions were taken for the body as each appeared.
    made = {makes.kind for makes in hers.makes.values()}
    numbers = run.presses.numbers_it_moves(TOLD_FROM_STANDING_PX * 2, moves.things)
    eligible = [thing for thing in moves.things.values() if thing.kind not in made and motion_was_measured(thing, at)
                and at - float(getattr(thing, "born", at)) >= old_for]
    exact = [thing for thing in eligible if thing.number in numbers]
    candidates = exact or [thing for thing in eligible if thing.kind in kinds]
    if len(candidates) == 1:
        hers.moved_by_her_presses(candidates[0], at, receipts=run.presses.entity_presses)


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

    def _the_one_of_a_kind_that_answers(self, moves: Any, at: float | None = None) -> int | None:
        """A known kind suggests a candidate whose own control receipts must answer."""
        from core.agency.which_one_answers_to_her import _goes_with_the_world, _how_the_world_went

        witnesses = []
        world = _how_the_world_went(self)
        for kind in self._by_kind:
            things = [t for t in moves.things.values() if t.kind == kind and t.moved and t.number not in self.not_mine
                      and motion_was_measured(t, getattr(t, 'seen', None) if at is None else at)]
            if len(things) != 1:
                continue
            number = things[0].number
            own = self._by_thing.get(number)
            witness = own_control_witness(number, own, epoch=self.identification.epoch,
                                           a_gauge=_a_gauge(self, number), movement_over=TOLD_FROM_STANDING_PX,
                                           with_the_world=own is not None and _goes_with_the_world(own, world))
            if witness is not None:
                witnesses.append(witness)
        return self.identification.choose(witnesses, visible=set(moves.things), established=None, excluded=self.not_mine)

    def moved_by_her_presses(self, thing: Any, at: float, *, receipts: Any = ()) -> None:
        """``thing`` is the one thing of a kind her presses move the same way every time: it is hers.

        The same evidence as her trials, read another way: a press is a trial,
        and a thing a key moves alike at every press answers to her. Offline
        2026-10-09 a runner whose number was split by a block during her
        trials was never found by them, while every press of space was seen
        to lift its kind and let it down.
        """
        if (self.number is not None or thing.number in self.not_mine or _a_gauge(self, thing.number)
                or not motion_was_measured(thing, at)):
            return
        from core.agency.which_one_answers_to_her import _goes_with_the_world, _how_the_world_went

        own = self._by_thing.get(thing.number)
        witness = own_control_witness(thing.number, own, epoch=self.identification.epoch, movement_over=TOLD_FROM_STANDING_PX,
                                       with_the_world=own is not None and _goes_with_the_world(own, _how_the_world_went(self)))
        if own is not None and _goes_with_the_world(own, _how_the_world_went(self)):
            return
        if witness is None or witness.statistic <= self.identification.statistic_over or witness.effect <= self.identification.effect_over:
            witness = own_trajectory_witness(thing, receipts, epoch=self.identification.epoch,
                statistic_over=self.identification.statistic_over, effect_over=self.identification.effect_over)
        if witness is None or self.identification.choose([witness], visible={thing.number}, established=None,
                                                         excluded=self.not_mine) != thing.number:
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


def _a_gauge(hers: Any, number: int) -> bool:
    """Whether a thing has filled and emptied from one end: a gauge her presses change, not her (core/agency/
    which_one_answers_to_her.py)."""
    outlines = getattr(hers, "_outlines", None)
    return bool(outlines is not None and outlines.a_gauge(number))


@invariant("agency.old_trials_need_a_current_measured_body", scope="agency",
           owner="core/agency/which_one_answers_to_her.py", observational=False)
def _current_body_motion_invariant() -> tuple:
    from types import SimpleNamespace

    from core.agency.which_one_answers_to_her import WhichIsHers

    hers = WhichIsHers()
    body = SimpleNamespace(number=1, kind=0, born=0., seen=9., x=50., y=50., size=100.,
                           vx=0., vy=0., w=10., h=10., moved=True,
                           motion_measured_at=None, motion_evidence='unavailable')
    for key, began, speed in (('a', 1., 60.), ('b', 3., -60.), ('a', 5., 60.), ('b', 7., -60.)):
        for i in range(2):
            hers._by_thing[1].add(key, speed, 0., press=began, position=(i * 30., 50.), when=began + .1 + i * .1)
    moves = SimpleNamespace(things={1: body})
    hers._decide(moves, 9.)
    assert hers.number is None
    body.motion_measured_at, body.motion_evidence = 9., 'measured'
    hers._decide(moves, 9.)
    assert hers.number == 1
    return ()
