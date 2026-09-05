"""tools/agi_gauntlet/as_she_sees_it.py — the sealed world, in her terms.

A gate that supplies its own policy measures the policy. The point of these
worlds is to measure her, so the world is presented in the shape her own
machinery already reads and the decision is taken by `look_ahead` —
the same lookahead that plays anything else she plays, scoring with the same
`how_good` and the same freedom term.

Two objects are all that takes.

``WhatSheHasWorkedOut`` is the model: it answers what a state would become
under an act, and how much of its own prediction has been holding. She builds
it by watching, and it is wrong at the start because nothing has told her
what the acts do.

``APlace`` is the state, in the vocabulary a situation is judged in: where
she is, what is still open from here, what is free. Nothing in it names this
world; a place with three exits is a place with three exits whatever produced
it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "APlace",
    "WhatSheHasWorkedOut",
    "forget_what_she_could_not_account_for",
    "play_as_she_would",
    "what_she_could_not_account_for",
    "what_would_account_for_it",
]


_UNEXPLAINED: dict[str, Any] = {}


def what_she_could_not_account_for(world: str) -> Any:
    """The situations her measure could not tell apart, in this world."""

    from core.agency.what_i_cannot_explain import WhatICannotExplain

    held = _UNEXPLAINED.get(world)
    if held is None:
        held = WhatICannotExplain()
        _UNEXPLAINED[world] = held
    return held


def forget_what_she_could_not_account_for(world: str = "") -> None:
    if world:
        _UNEXPLAINED.pop(world, None)
    else:
        _UNEXPLAINED.clear()


def what_would_account_for_it(world: str, *, most: int = 3) -> list:
    """The properties best worth trying, from her own unexplained pairs.

    Not the answer. Explaining what already happened and improving what
    happens next are different things, and the second is decided by a trial
    in the life rather than by a fit to the past.
    """

    return what_she_could_not_account_for(world).worth_trying(most=most)


@dataclass(frozen=True)
class APlace:
    """A state in the terms a situation is read in. No world in it."""

    where: tuple[int, int]
    size: int
    #: The number the world shows. Nothing says what it means.
    score: float = 0.0
    #: Places she has already been.
    seen: frozenset = frozenset()
    #: Places she has learned end the run.
    fatal: frozenset = frozenset()
    over: bool = False

    def as_text(self) -> str:
        return f"{self.where[0]},{self.where[1]}"

    def newness(self) -> float:
        """Whether she has been here before, and how much is unseen beyond."""

        here = 0.0 if self.where in self.seen else 1.0
        around = self.empty() / 4.0
        return max(0.0, min(1.0, 0.5 * here + 0.5 * around))

    def observations(self) -> list[tuple[float, float, int, int]]:
        """What this place says about itself, for a measure she composes.

        One observation per square she can see from here: how new it is, and
        where it sits. Her measure-invention algebra reads observations rather
        than boards now, so a place-world can feed it — and what it can then
        compose over them includes the property her authored measures cannot
        express here, which is that somewhere she has not been is worth more
        than somewhere she has.
        """

        seen: list[tuple[float, float, int, int]] = []
        for x in range(self.size):
            for y in range(self.size):
                place = (x, y)
                if place in self.fatal:
                    continue
                new = 0.0 if place in self.seen else 1.0
                near = 1.0 / (
                    1.0 + abs(place[0] - self.where[0]) + abs(place[1] - self.where[1])
                )
                seen.append((new, near, x, y))
        return seen

    def numbers(self) -> tuple[float, ...]:
        """The number the world shows, in the units her measure reads.

        Nearness counts doublings, because it was written for a world where
        doubling is the step that matters. Handed a reading that grows by one
        at a time it is nearly flat — 0.977 against 0.983 for the last two
        steps of a climb — and a gradient that small loses to every other
        term. Presented as a power of two it comes back linear, which is what
        this world's reading actually is.

        The coordinates are deliberately not here. Where she is is not how
        she is doing, and scoring a state by its coordinates rewards a corner
        of the grid for being a corner.
        """

        return (2.0 ** min(48.0, max(0.0, float(self.score))),)

    def empty(self) -> int:
        """How much is still open FROM HERE, which is what room means.

        Two wrong readings came before this one. Counting every unvisited
        square made exploring look like filling a board: each new square
        lowered the term, so her judgement preferred standing still. Counting
        every square not known fatal made the term the same everywhere, so it
        said nothing at all, and she climbed the reading to a ridge and then
        oscillated between two squares forever, each move as good as the
        other because nothing could tell them apart.

        Room is local. A place with unexplored neighbours has room; a place
        whose neighbours are all visited or all fatal has none, and that is
        the difference between the frontier and a dead end.

        Named ``empty`` because that is what her room term reads. Called
        ``free`` it was never read at all, and room was nought everywhere for
        the whole of the first measurement.
        """

        around = 0
        for step in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            there = (self.where[0] + step[0], self.where[1] + step[1])
            if not (0 <= there[0] < self.size and 0 <= there[1] < self.size):
                continue
            if there in self.fatal or there in self.seen:
                continue
            around += 1
        return around

    def places(self) -> int:
        """What room is measured against: the ways out of one place."""

        return 4


@dataclass
class WhatSheHasWorkedOut:
    """Her model of what the acts do, and how far it has been holding.

    ``expect`` is the whole interface her lookahead needs. ``confidence`` is
    the share of her own predictions that came true, which is what discounts
    a deep future — a model that has been wrong makes a five-move plan worth
    less than a one-move one, and that has to be measured rather than set.
    """

    effects: dict[str, tuple[int, int]] = field(default_factory=dict)
    fatal: set = field(default_factory=set)
    predicted: int = 0
    held: int = 0
    tried: dict = field(default_factory=dict)
    #: What the number read at each place she has stood.
    readings: dict = field(default_factory=dict)

    def expect(self, state: APlace, act: str) -> APlace | None:
        step = self.effects.get(str(act))
        if step is None:
            return None
        there = (
            min(state.size - 1, max(0, state.where[0] + step[0])),
            min(state.size - 1, max(0, state.where[1] + step[1])),
        )
        if there in self.fatal:
            # Her model says this act leads nowhere. Returning a bad state
            # instead of nothing let the lookahead weigh it against the
            # others and sometimes take it, and her freedom term counts an
            # act leading nowhere as no option at all — which is the whole
            # instruction the term exists to encode. She walked back into the
            # same square that had already ended a run, on move one, in every
            # episode.
            return None
        return APlace(
            where=there,
            size=state.size,
            score=self.expected_score(there, state),
            # Where she has been BEFORE arriving, not including where she has
            # just arrived. Adding the destination made every imagined future
            # a place she had already been, so the newness term read nought
            # for all of them and the one property that could tell going
            # somewhere from pacing was dead in every comparison it was
            # written for.
            seen=state.seen,
            fatal=state.fatal,
        )

    def expected_score(self, there: tuple[int, int], state: APlace) -> float:
        """What the number would read there, from what it has read before.

        She has no rule for it and does not need one: the places she has
        stood carry the readings she got, and an unvisited place is guessed
        from the nearest one she has, which is a guess her own confidence
        term then discounts.
        """

        held = self.readings.get(there)
        if held is not None:
            return float(held)
        if not self.readings:
            return float(state.score)
        # The tightest guess consistent with everything she has read, rather
        # than the reading nearest by. A number that falls by at most ``slope``
        # a step cannot be lower than any reading minus slope times the
        # distance to it, so the best guess is the largest of those bounds —
        # and every reading she has contributes to it.
        #
        # Taking the nearest one alone throws the rest away, and far from
        # anywhere she has stood that is a guess made from one number. She
        # climbed correctly near what she had seen and wandered beyond it.
        #
        # ``slope`` used to be one, which is true of a number that counts the
        # steps left to the goal and of nothing else. Stated that way it is a
        # prior about the world supplied on her behalf: it says the reading
        # points at the answer, which is exactly the thing a world with no
        # instructions is supposed to be withholding. It is measured now, from
        # the readings she has taken at places a step apart. A signal that does
        # not move with position measures nought, the bound goes flat, and
        # nothing she has read pretends to say where to go — which is the
        # correct reading of a number that is not a gradient.
        rate = self.how_fast_it_changes()
        if rate <= 0.0:
            return sum(self.readings.values()) / len(self.readings)
        return max(
            float(value)
            - rate * (abs(place[0] - there[0]) + abs(place[1] - there[1]))
            for place, value in self.readings.items()
        )

    def how_fast_it_changes(self) -> float:
        """The most the reading has been seen to move over one step.

        Empirical and one-sided: the largest change observed between two
        places a step apart. Anything smaller would make the bound below claim
        more than the readings support, and the bound is only useful because
        nothing she has seen contradicts it.
        """

        fastest = 0.0
        for place, value in self.readings.items():
            for step in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                beside = self.readings.get((place[0] + step[0], place[1] + step[1]))
                if beside is None:
                    continue
                moved = abs(float(beside) - float(value))
                if moved > fastest:
                    fastest = moved
        return fastest

    def confidence(self) -> float:
        """No prediction, no confidence. Nothing is assumed on her behalf."""

        return (self.held / self.predicted) if self.predicted else 0.0

    def watched(
        self,
        state: APlace,
        act: str,
        landed: tuple[int, int],
        died: bool,
        score: float = 0.0,
    ) -> None:
        """One act, and what it actually did. The only way the model grows."""

        self.readings[landed] = float(score)

        thought = self.expect(state, act)
        if thought is not None:
            # Only a prediction that was made counts. Counting the acts she
            # had no model for as failed predictions held her confidence at
            # nought, and her lookahead refuses to plan on a model with no
            # confidence — so she could not act without confidence and could
            # not earn confidence without acting.
            self.predicted += 1
            if thought.where == landed:
                self.held += 1
        self.tried[str(act)] = self.tried.get(str(act), 0) + 1
        self.effects[str(act)] = (
            landed[0] - state.where[0],
            landed[1] - state.where[1],
        )
        if died:
            self.fatal.add(landed)


def play_as_she_would(
    world: Any,
    knows: WhatSheHasWorkedOut,
    *,
    budget: int,
    budget_s: float = 0.05,
    learn: bool = True,
) -> dict[str, Any]:
    """Play one episode with her lookahead, learning as it goes.

    An act she has never taken has no entry in the model, so her lookahead
    returns nothing for it and she takes it to find out — which is the same
    decision the expected-information-gain policy makes about an observation,
    arrived at from the other side.
    """

    from core.agency.how_good_is_this import (
        AS_GOOD_A_GUESS_AS_ANY,
        how_good,
        terms,
    )
    from core.agency.looking_ahead import look_ahead
    from core.agency.what_matters_here import for_this_world

    mattered = for_this_world(str(getattr(world, "name", "somewhere")) if learn else "")
    weights = mattered.weights(AS_GOOD_A_GUESS_AS_ANY) if learn else None
    along: list[dict[str, float]] = []
    #: The situations she passed through, kept so that what her measure
    #: could not account for can be looked at afterwards.
    passed: list[Any] = []
    world.reset()
    said = world.look()
    seen = {said["where"]}
    knows.readings[said["where"]] = float(said.get("score", 0.0))
    # What she is aiming at: the highest reading she has seen, plus room for
    # a higher one. Nothing tells her the number has a maximum, so what she
    # steers by is "more than the best so far", which is what makes this
    # goal discovery rather than goal following.
    while world.moves < budget and not (world.won or world.lost):
        said = world.look()
        here = APlace(
            where=said["where"],
            size=said["size"],
            score=float(said.get("score", 0.0)),
            seen=frozenset(seen),
            fatal=frozenset(knows.fatal),
        )
        # What she is steering at: one step better than the best reading she
        # has seen. She has not been told the number has a top, so what she
        # can aim at is "better than anything so far", which is what makes
        # this finding the goal rather than following it.
        best = max([*knows.readings.values(), here.score]) + 1
        toward = f"{2.0 ** min(48.0, max(0.0, best)):.0f}"
        unknown = [act for act in world.acts if act not in knows.effects]
        if unknown:
            act = unknown[0]
        elif knows.confidence() <= 0.0:
            # She has a model and no evidence that it holds, and her lookahead
            # correctly refuses to plan on that. Testing the least-tested act
            # is what earns the confidence — the same decision the
            # information-gain policy makes about an observation, reached
            # from the other side.
            act = min(world.acts, key=lambda name: knows.tried.get(name, 0))
        else:
            along.append(terms(here, toward=toward, knows=knows, acts=world.acts))
            passed.append(
                (here, how_good(here, toward=toward, weights=weights,
                                knows=knows, acts=world.acts))
            )
            ranked = look_ahead(
                knows, here, list(world.acts), toward=toward,
                budget_s=budget_s, weights=weights,
            )
            if not ranked:
                act = min(world.acts, key=lambda name: knows.tried.get(name, 0))
            else:
                act = max(ranked, key=lambda name: ranked[name][0])
        before = here
        landed = world.do(act)
        knows.watched(
            before, act, landed["where"], world.lost,
            score=float(landed.get("score", 0.0)),
        )
        seen.add(landed["where"])
    if learn:
        mattered.watched(along, went_well=world.won)
        # How each situation turned out: the best reading the run reached
        # after it, against the best she has ever seen here.
        #
        # Not whether it was won. With no win yet every outcome is nought,
        # every pair turns out the same, there is nothing her measure failed
        # to account for, and the one mechanism that could break the deadlock
        # is switched off by the deadlock. How high the reading got is a real
        # difference between two runs that both ended badly.
        cannot_explain = what_she_could_not_account_for(
            str(getattr(world, "name", "somewhere"))
        )
        readings = [one.score for one, _s in passed] or [0.0]
        ceiling = max([*knows.readings.values(), *readings]) or 1.0
        for index, (situation, scored) in enumerate(passed):
            after = max(readings[index:]) if index < len(readings) else 0.0
            cannot_explain.been_here(
                situation, scored, (after / ceiling) if ceiling else 0.0
            )
    return {
        "won": world.won,
        "moves": world.moves,
        "lost": world.lost,
        "weights_moved": mattered.what_it_learned()["moved"] if learn else False,
    }
