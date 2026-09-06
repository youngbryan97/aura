"""core/cognition/does_improving_compound.py — is there a second generation.

Three things get called self-improvement and only the third is the claim
people mean.

Level 0 is adaptation: change a policy or a representation while the learning
machinery stays fixed. Level 1 is meta-adaptation: change some part of the
machinery that performs or evaluates adaptation. Aura does both, and
`she_improves_her_own_deciding` is the second — it searches for a rule that
would have chosen better than the rule in force, and installs it.

Level 2 is the one that is not established:

    M₀ --I_{M₀}--> M₁ --I_{M₁}--> M₂ --I_{M₂}--> M₃ ...

with Q(M_{n+1}) > Q(M_n), where each M_{n+1} changes the future improvement
operator rather than solving another object-level task. One arrow is
self-improvement. A sequence of them, each conducted by the product of the
last, is compounding, and nobody had run the sequence.

This runs it. What it reports is what happened, including that it did not
compound, because a generational study whose only possible outcome is
"it compounds" is a demonstration rather than a measurement.

Three disciplines, and each of them is a way the result could have been fake.

**Q is measured on families the search never saw.** The improver reads the
record to find a better rule, so scoring it on the same record scores it on
its own training set, and every generation would look like an improvement
until the day it was used. The families are split, the search sees one half,
Q is the other half, and the split is by name so it does not drift between
generations.

**Generation n+1 is searched by M_n.** Otherwise the chain is n independent
searches of the same space with the same operator, plotted in a row, and the
line goes up for the reason any repeated search does. What makes it a chain
is that the operator conducting each search is the previous product.

**A generation that changes nothing is the end of the chain.** An operator
identical to its parent has not changed the future improvement operator, so
the sequence stopped whether or not the numbers kept moving.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from core.cognition.the_floor_she_stands_on import Code, OutOfFuel, Stuck, every_code

logger = logging.getLogger("Aura.Cognition.DoesImprovingCompound")

__all__ = [
    "Chain",
    "HOW_MANY_REPLAYS",
    "WHILE_SEARCHING",
    "spread_of",
    "a_replay_that_changes_nothing",
    "Generation",
    "Verdict",
    "against_its_null",
    "held_out_cost",
    "the_generations",
    "what_the_split_is",
]


#: How the record is cut in two. By a hash of the family name rather than by
#: when it was seen, so the same family lands on the same side at every
#: generation — a split that drifts turns a held-out score into a different
#: score each round, and the comparison between generations stops meaning
#: anything.
def what_the_split_is(family: str) -> str:
    """Which half of the record this family belongs to. Stable, by name."""

    total = 0
    for character in str(family):
        total = (total * 31 + ord(character)) & 0xFFFFFFFF
    return "searched" if total % 2 == 0 else "held out"


def _families_on(side: str) -> list[str]:
    from core.cognition.the_record_of_her_own_work import the_record

    return sorted(
        {
            one.family
            for one in the_record().kept
            if what_the_split_is(one.family) == side
        }
    )


@contextmanager
def a_replay_that_changes_nothing(seed: int | None = None) -> Iterator[None]:
    """Hold still everything the pricing reads and the replay writes.

    The replay is supposed to be a measurement: ask what a rule would have
    chosen, and charge what that choice actually cost when it was made. The
    docstring on the shipped version says a rule cannot flatter itself by
    changing the measurements, because what is recorded is what the search
    spent and the search does not read it.

    It could. Asking what to do next writes a stage into the decision trace
    and an entry into the ledger of what each action has done, and the
    pricing reads that ledger. So two identical replays of the same rule over
    the same record returned 45,408 and then 42,537 — the second one cheaper
    because the first had happened. Every generational comparison built on
    that is comparing a rule against a moving target, and a rule evaluated
    more often looks better for having been evaluated.

    Snapshot, run, restore. The scope covers the trace, the done-ledger and
    the whole record — not only its episodes, since restoring `kept` alone
    left `uses` growing and the risk term reads how many entries there are.

    The draw is the other half. Deciding what to try among actions with no
    history is a draw from the Beta their counts imply, which is the right
    mechanism and is stochastic, so one replay of a rule is one sample of it.
    Seeding here makes a replay repeatable; the caller averages over seeds,
    because a single sample of a stochastic policy cannot be compared with
    another single sample and one sample never settles whether a difference
    is real.
    """

    import copy
    import random

    from core.cognition import she_decides_to_develop as deciding
    from core.cognition.the_record_of_her_own_work import the_record
    from core.cognition.what_she_could_do_next import WHAT_THEY_HAVE_DONE

    held = the_record()
    trace = list(deciding._TRACE)
    done = copy.deepcopy(dict(WHAT_THEY_HAVE_DONE))
    # The whole record, not only its episodes. Restoring `kept` alone left
    # `uses` growing across replays, the risk term reads how many entries
    # there are, and the price of every action drifted down the more often it
    # was priced: 51,692 then 40,631 then 30,027 for the same rule over the
    # same record. A rule looked better for having been considered before.
    was = (
        list(held.kept),
        copy.deepcopy(held.families),
        copy.deepcopy(held.uses),
        dict(held.last_used),
        held.seen,
    )
    where = random.getstate()
    if seed is not None:
        random.seed(seed)
    try:
        yield
    finally:
        random.setstate(where)
        deciding._TRACE[:] = trace
        WHAT_THEY_HAVE_DONE.clear()
        WHAT_THEY_HAVE_DONE.update(done)
        held.kept[:] = was[0]
        held.families = was[1]
        held.uses = was[2]
        held.last_used = was[3]
        held.seen = was[4]


#: Replays averaged for one Q. A stochastic policy sampled once gives a
#: number that moves by more than the differences being measured: the same
#: rule over the same record came back 51,692 and 30,027.
HOW_MANY_REPLAYS = 7

#: Replays per candidate while searching. Fewer, deliberately: a search that
#: proposes a lucky candidate costs one wasted proposal, and the held-out
#: score refuses it. Spending the verdict's precision on every candidate
#: spends the whole budget on the candidates that were never going to win.
WHILE_SEARCHING = 2


def held_out_cost(
    worth: Code, *, side: str = "held out", samples: int = HOW_MANY_REPLAYS
) -> float:
    """Q, as the mean over seeded replays. See ``spread_of`` for the noise."""

    got = _replays(worth, side=side, samples=samples)
    return sum(got) / len(got) if got else float("inf")


def spread_of(
    worth: Code, *, side: str = "held out", samples: int = HOW_MANY_REPLAYS
) -> float:
    """How much Q moves between seeds. An improvement inside this is noise."""

    got = _replays(worth, side=side, samples=samples)
    if len(got) < 2:
        return float("inf")
    mean = sum(got) / len(got)
    return (sum((one - mean) ** 2 for one in got) / (len(got) - 1)) ** 0.5


def _replays(worth: Code, *, side: str, samples: int) -> list[float]:
    got = [_one_replay(worth, side=side, seed=seed) for seed in range(max(1, samples))]
    return [one for one in got if one != float("inf")]


def _one_replay(worth: Code, *, side: str = "held out", seed: int = 0) -> float:
    """What this rule would have cost on families the search cannot see.

    The same replay `she_improves_her_own_deciding` does, restricted to one
    side of the split. A rule found by reading half the record and scored on
    the other half is being asked the only question that matters about it:
    does it generalise, or has it memorised an afternoon.
    """

    from core.cognition.the_record_of_her_own_work import the_record
    from core.cognition.what_it_is_worth_doing import (
        the_worth_she_uses,
        the_worth_she_wrote,
    )
    from core.cognition.what_she_could_do_next import the_actions_she_has

    kept = the_record().kept
    if not kept or not the_actions_she_has():
        return float("inf")
    wanted = set(_families_on(side))
    if not wanted:
        return float("inf")
    cost_of: dict[tuple[str, str | None], list[int]] = {}
    for one in kept:
        cost_of.setdefault((one.family, one.route), []).append(one.walked)
        # An action that was tried and did not hold still cost what it cost.
        # Keying only on `route` meant the replay could price a chosen action
        # only where that action had once WORKED, so a policy that picks
        # something never yet successful was charged the family's average
        # instead — the same number for every operator.
        acted = getattr(one, "tried", None)
        if acted and acted != one.route:
            cost_of.setdefault((one.family, acted), []).append(one.walked)
    was = the_worth_she_uses()
    total = 0.0
    try:
        the_worth_she_wrote(worth)
        with a_replay_that_changes_nothing(seed=seed):
            total = _replay(sorted(wanted), cost_of, kept)
    except (OutOfFuel, Stuck, ArithmeticError, TypeError, ValueError):
        return float("inf")
    finally:
        the_worth_she_wrote(was)
    return total


def _replay(
    families: list[str],
    cost_of: dict[tuple[str, str | None], list[int]],
    kept: list[Any],
) -> float:
    from core.cognition.she_decides_to_develop import what_to_do_next

    total = 0.0
    for family in families:
        spent = cost_of.get((family, None)) or [
            one.walked for one in kept if one.family == family
        ]
        now = sum(spent) / len(spent)
        decided = what_to_do_next(family, costs_now=int(now))
        picked = decided.action.name if decided.action else None
        seen = cost_of.get((family, picked))
        total += (sum(seen) / len(seen)) if seen else now
    return total


@dataclass(frozen=True)
class Generation:
    """One link in the chain, and what it is worth on ground it never saw."""

    index: int
    #: The operator this generation produced.
    operator: Any
    #: The operator that conducted the search producing it. None for M₀.
    searched_by: Any = None
    #: Q, on the held-out half.
    quality: float = float("inf")
    #: Q of the operator that produced this one, so the comparison is local.
    parent_quality: float = float("inf")
    #: How far two means of this size can differ by the draw alone. A fall
    #: inside it is the sampling, not the operator.
    noise: float = 0.0
    searched_for: float = 0.0

    @property
    def improved(self) -> bool:
        """Lower cost on ground the search never saw, by more than the noise.

        The bar is not "lower". Deciding what to try among actions with no
        history is a draw, so Q is the mean of a stochastic quantity, and the
        same rule over the same record measured 51,692 and 30,027 on single
        replays. An improvement smaller than that says nothing, and a chain
        built out of such improvements is a plot of the sampling error.
        """
        return (self.parent_quality - self.quality) > self.noise

    @property
    def changed_the_operator(self) -> bool:
        """Whether this is a new operator at all, rather than its parent again."""
        return self.index == 0 or repr(self.operator) != repr(self.searched_by)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "operator": repr(self.operator)[:200],
            "quality": None if self.quality == float("inf") else round(self.quality, 3),
            "parent_quality": (
                None if self.parent_quality == float("inf") else round(self.parent_quality, 3)
            ),
            "noise": round(self.noise, 3),
            "improved": self.improved,
            "changed_the_operator": self.changed_the_operator,
            "searched_for": round(self.searched_for, 2),
        }


@dataclass(frozen=True)
class Chain:
    """What happened when the sequence was actually run."""

    generations: tuple[Generation, ...] = ()
    stopped_because: str = ""
    #: Whether each generation was searched by its parent. False is the null.
    chained: bool = True
    searched_families: tuple[str, ...] = field(default_factory=tuple)
    held_out_families: tuple[str, ...] = field(default_factory=tuple)

    @property
    def depth(self) -> int:
        """How many times an operator produced a better operator, in a row.

        Zero is level one: a self-improvement happened and nothing followed
        from it. Two or more is the claim.
        """
        depth = 0
        for one in self.generations[1:]:
            if not (one.improved and one.changed_the_operator):
                break
            depth += 1
        return depth

    @property
    def compounds(self) -> bool:
        return self.depth >= 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "compounds": self.compounds,
            "generations": [one.to_dict() for one in self.generations],
            "chained": self.chained,
            "stopped_because": self.stopped_because,
            "searched": list(self.searched_families),
            "held_out": list(self.held_out_families),
        }


def _a_better_operator(
    under: Code, *, deepest: int, within: float
) -> tuple[Code | None, float]:
    """Search the floor for a rule the SEARCHED half says beats this one.

    Conducted under ``under``: the operator in force while the search runs is
    the previous generation's product, which is what makes this a chain
    rather than a row of independent searches.
    """

    from core.cognition.what_it_is_worth_doing import (
        the_worth_she_uses,
        the_worth_she_wrote,
    )

    was = the_worth_she_uses()
    began = time.monotonic()
    try:
        the_worth_she_wrote(under)
        # The search may be noisy; the verdict may not. A proposer that draws
        # a lucky sample costs a wasted candidate, and the held-out score
        # then refuses it. Averaging seven replays per candidate instead buys
        # precision where it is not needed and spends the whole budget doing
        # it — twelve candidates examined instead of a hundred.
        best = held_out_cost(under, side="searched", samples=WHILE_SEARCHING)
        if best == float("inf"):
            return None, best
        found: Code | None = None
        for candidate in every_code(
            deepest=deepest, variables=4, constants=(0, 1, 2), also=()
        ):
            if time.monotonic() - began >= within:
                break
            closed: Code = candidate
            for _ in range(4):
                closed = Code("given a thing", parts=(closed,))
            try:
                got = held_out_cost(closed, side="searched", samples=WHILE_SEARCHING)
            except (OutOfFuel, Stuck, ArithmeticError, TypeError, ValueError):
                continue
            if got < best:
                best, found = got, closed
        return found, best
    finally:
        the_worth_she_wrote(was)


def the_generations(
    *, how_many: int = 3, deepest: int = 2, within: float = 8.0, chained: bool = True
) -> Chain:
    """Run the sequence and report what it did, including nothing.

    M₀ is the rule in force. Each later generation is searched by the one
    before it, scored on the half of the record no search reads, and compared
    with its own parent rather than with M₀ — because a chain is a series of
    local improvements, and comparing every generation with the first hides a
    generation that made things worse after one that helped.

    ``chained=False`` is the null and it is the whole argument. Every
    generation is searched by M₀ instead of by its parent, so the sequence
    becomes n independent searches of the same space by the same operator,
    with the products still installed in order. If the numbers fall just as
    far under the null, the improvement is what any repeated search of a
    space gives you and the chaining contributed nothing — and a study with
    no null is a plot of a line going up.
    """

    from core.cognition.sequence_induction import _register_what_she_could_do
    from core.cognition.what_it_is_worth_doing import THE_WORTH
    from core.cognition.what_she_could_do_next import the_actions_she_has

    # The chain needs something to choose between. With no actions registered
    # every replay costs infinity, every generation ties, and the study
    # reports "it does not compound" for a reason that has nothing to do with
    # whether it compounds.
    if not the_actions_she_has():
        try:
            _register_what_she_could_do()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("could not register what she could do: %s", exc)
    if not the_actions_she_has():
        return Chain(
            stopped_because=(
                "no developmental actions are registered; there is nothing for "
                "an operator to be better at choosing between"
            )
        )

    searched = tuple(_families_on("searched"))
    held = tuple(_families_on("held out"))
    if not searched or not held:
        return Chain(
            stopped_because=(
                f"the record splits {len(searched)} searched / {len(held)} held "
                "out; both halves have to have families before a held-out score "
                "is a score"
            ),
            searched_families=searched,
            held_out_families=held,
        )

    generations = [
        Generation(
            index=0,
            operator=THE_WORTH,
            quality=held_out_cost(THE_WORTH),
            parent_quality=float("inf"),
            noise=spread_of(THE_WORTH) / (HOW_MANY_REPLAYS**0.5),
        )
    ]
    stopped = ""
    for index in range(1, max(1, how_many)):
        parent = generations[-1]
        conducted_by = parent.operator if chained else generations[0].operator
        began = time.monotonic()
        found, _ = _a_better_operator(
            conducted_by, deepest=deepest, within=within
        )
        spent = time.monotonic() - began
        if found is None:
            stopped = (
                f"generation {index}: nothing in the floor beat M{index - 1} on "
                "the searched half within the budget"
            )
            break
        found_quality = held_out_cost(found)
        generations.append(
            Generation(
                index=index,
                operator=found,
                searched_by=conducted_by,
                quality=found_quality,
                parent_quality=parent.quality,
                # Two means of HOW_MANY_REPLAYS samples each. The difference
                # of two means has the wider spread of the two, so the bar is
                # the larger standard error times root two.
                noise=(
                    max(spread_of(found), spread_of(parent.operator))
                    / (HOW_MANY_REPLAYS**0.5)
                )
                * (2**0.5),
                searched_for=spent,
            )
        )
        if not generations[-1].changed_the_operator:
            stopped = f"generation {index} is its own parent; the chain ended"
            break
        if not generations[-1].improved:
            stopped = (
                f"generation {index} was better on the half it was found from "
                "and not on the half it was not; that is memorisation, not "
                "improvement"
            )
            break
    return Chain(
        generations=tuple(generations),
        stopped_because=stopped,
        chained=chained,
        searched_families=searched,
        held_out_families=held,
    )


@dataclass(frozen=True)
class Verdict:
    """The chain against its own null, which is the only form of the claim.

    A chain that reaches depth three, next to an unchained control that
    reaches depth three, says that searching a space repeatedly finds better
    members of it. That is true of any search and it is not recursive
    improvement. The claim needs the chained run to get somewhere the
    unchained one does not.
    """

    chain: Chain
    null: Chain

    @property
    def deeper_than_its_null(self) -> bool:
        return self.chain.depth > self.null.depth

    @property
    def better_than_its_null(self) -> float:
        """How much further Q fell, chained minus unchained. Positive is the
        chain doing something."""

        def fell(one: Chain) -> float:
            live = [g for g in one.generations if g.quality != float("inf")]
            if len(live) < 2:
                return 0.0
            return live[0].quality - live[-1].quality

        return fell(self.chain) - fell(self.null)

    @property
    def holds(self) -> bool:
        """Level two, or not. Both halves, and the null is half of it."""
        return (
            self.chain.compounds
            and self.deeper_than_its_null
            and self.better_than_its_null > 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "holds": self.holds,
            "deeper_than_its_null": self.deeper_than_its_null,
            "further_than_its_null": round(self.better_than_its_null, 3),
            "chain": self.chain.to_dict(),
            "null": self.null.to_dict(),
        }


def against_its_null(
    *, how_many: int = 4, deepest: int = 3, within: float = 8.0
) -> Verdict:
    """Run the chain and the control, and report both."""

    return Verdict(
        chain=the_generations(
            how_many=how_many, deepest=deepest, within=within, chained=True
        ),
        null=the_generations(
            how_many=how_many, deepest=deepest, within=within, chained=False
        ),
    )
