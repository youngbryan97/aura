"""When several readings survive, which one thing should she try next.

The old form of this could only compare two readings of a sequence, over an
enumeration of sequence states written into it. That is the same question in a
costume: several accounts survive the evidence, and she has to pick the one act
whose outcome cuts the most of them away.

Nothing about that is particular to sequences. It needs three things and no
others: the accounts, the acts she could perform, and a way to ask an account
what it expects from an act. Sequences supply those. So do a board, a screen, a
question she could put to him, a measurement she has not taken.

Three answers come out of it.

The act to perform is the one that leaves the least still unsettled per unit of
what it costs — the standard measure, with every quantity in it read off the
accounts rather than chosen.

What an outcome ruled out is then arithmetic: an account that expected
something else is gone.

And when no act separates two accounts, that is not a failure to find one. It
is the discovery that in this world they are the same account, and the pair
belongs to :mod:`core.cognition.one_thing_many_spellings` rather than to the
search. A set that cannot be narrowed further has been settled as far as the
world admits, which is a different thing from being undecided.

That last answer is only ever as wide as the acts she was given, and this was
worth finding out the hard way: over states of length two and four, two of her
readings agreed everywhere and looked like one reading wearing two names. Add
lengths three and five and they part on the first try. So the question the
third answer settles is "can these acts separate them", and it is named for
that. An act space that leaves something out reports a distinction the world
makes as a distinction it does not, which is worse than reporting nothing, and
no caller should be able to read the answer as absolute by accident.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Experiment",
    "WhatTheseActsSaw",
    "every_act_that_settles_a_sequence",
    "how_many_would_settle_it",
    "what_it_ruled_out",
    "what_these_acts_cannot_separate",
    "what_to_try",
]

#: What an account says when it will not answer for an act. Distinct from any
#: answer it could give, including None.
SILENT = object()


@dataclass(frozen=True)
class Experiment:
    """One act, and what performing it would do to the field of accounts."""

    #: The act itself, in whatever terms the caller speaks.
    do: Any
    #: What each account expects, by name. Absent means it would not say.
    expects: Mapping[str, Any]
    #: How many different answers the surviving accounts predict.
    tells_apart: int
    #: Bits of doubt before performing it.
    unsettled_now: float
    #: Bits of doubt expected to remain after.
    unsettled_after: float
    #: What performing it costs, in the caller's units.
    costs: float
    #: Accounts that would say nothing either way.
    silent: tuple[str, ...] = field(default=())

    @property
    def settles(self) -> float:
        """Bits it is expected to settle."""
        return self.unsettled_now - self.unsettled_after

    @property
    def worth_doing(self) -> float:
        """Bits settled per unit of cost. What the choice is made on."""
        return self.settles / self.costs if self.costs > 0 else math.inf

    def __str__(self) -> str:
        answers = sorted({_readable(one) for one in self.expects.values()})
        return (
            f"try {self.do!r}: {self.tells_apart} different answers "
            f"({', '.join(answers)}), settling {self.settles:.2f} of "
            f"{self.unsettled_now:.2f} bits"
        )


def _readable(answer: Any) -> str:
    try:
        return str(list(answer)) if isinstance(answer, tuple) else str(answer)
    except Exception:  # noqa: BLE001 - last-resort floor: str() of a hostile object
        return repr(answer)


def _how_likely_before_any_of_it(
    hypotheses: Mapping[str, Any],
    plausibility: Callable[[str, Any], float] | None,
) -> dict[str, float]:
    """What she believed before this act, normalised.

    Given nothing else, shorter accounts are likelier, which is not a taste but
    the count of what could have been said instead: there are twice as many
    accounts one symbol longer. When an account will not say how long it is,
    the honest weight is the same as its neighbours'.
    """
    weights: dict[str, float] = {}
    if plausibility is not None:
        for name, one in hypotheses.items():
            weights[name] = max(0.0, float(plausibility(name, one)))
    else:
        lengths = {name: _how_long(one) for name, one in hypotheses.items()}
        measured = [n for n in lengths.values() if n is not None]
        # An account that will not say how long it is weighs like its
        # neighbours. Reading it as length zero made it the LIKELIEST account
        # in the set, because every account that DID answer is at least one
        # symbol long and so weighs at most a half.
        typical = sum(measured) / len(measured) if measured else 1.0
        for name, length in lengths.items():
            weights[name] = 2.0 ** -(typical if length is None else length)
    total = sum(weights.values())
    if total <= 0:
        return {name: 1.0 / len(hypotheses) for name in hypotheses}
    return {name: weight / total for name, weight in weights.items()}


def _how_long(one: Any) -> int | None:
    """How many symbols the account takes, or None when it will not say.

    None and zero are different answers and the caller weighs them
    differently, so this must not collapse one into the other.
    """
    from core.cognition.what_it_costs_to_say import how_long_it_is

    try:
        length = int(how_long_it_is(one))
    except Exception:  # noqa: BLE001 - accounts are hers and may be anything
        return None
    return max(0, length)


def _doubt(weights: Mapping[str, float]) -> float:
    """Bits of doubt across a field of accounts."""
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    bits = 0.0
    for weight in weights.values():
        share = weight / total
        if share > 0:
            bits -= share * math.log2(share)
    return bits


def _what_each_expects(
    hypotheses: Mapping[str, Any],
    act: Any,
    predicts: Callable[[Any, Any], Any],
) -> dict[str, Any]:
    """Every account's expectation of one act. Refusing to say is an answer."""
    said: dict[str, Any] = {}
    for name, one in hypotheses.items():
        try:
            answer = predicts(one, act)
        except Exception:  # noqa: BLE001 - refusing to say IS an answer here
            answer = SILENT
        said[name] = SILENT if answer is None else answer
    return said


def _key(answer: Any) -> Any:
    """Something hashable that two equal answers share."""
    if answer is SILENT:
        return SILENT
    try:
        hash(answer)
    except TypeError:
        return repr(answer)
    return answer


def _weigh_one_act(
    hypotheses: Mapping[str, Any],
    act: Any,
    predicts: Callable[[Any, Any], Any],
    before: Mapping[str, float],
    costs: float,
) -> Experiment:
    """What one act would leave unsettled.

    An account that would not answer is not evidence either way, so it survives
    whatever happens — which is why an act that silences half the field scores
    badly without anything needing to say so.
    """
    said = _what_each_expects(hypotheses, act, predicts)
    silent = tuple(sorted(name for name, one in said.items() if one is SILENT))
    speaking = {name: one for name, one in said.items() if one is not SILENT}

    outcomes: dict[Any, list[str]] = {}
    for name, answer in speaking.items():
        outcomes.setdefault(_key(answer), []).append(name)

    unsettled_now = _doubt(before)
    if not outcomes:
        return Experiment(
            do=act,
            expects={},
            tells_apart=0,
            unsettled_now=unsettled_now,
            unsettled_after=unsettled_now,
            costs=costs,
            silent=silent,
        )

    after = 0.0
    for names in outcomes.values():
        chance = sum(before.get(name, 0.0) for name in names)
        if chance <= 0:
            continue
        surviving = {name: before.get(name, 0.0) for name in [*names, *silent]}
        after += chance * _doubt(surviving)

    return Experiment(
        do=act,
        expects=dict(speaking),
        tells_apart=len(outcomes),
        unsettled_now=unsettled_now,
        unsettled_after=after,
        costs=costs,
        silent=silent,
    )


def what_to_try(
    hypotheses: Mapping[str, Any],
    could_try: Iterable[Any],
    *,
    predicts: Callable[[Any, Any], Any],
    plausibility: Callable[[str, Any], float] | None = None,
    costs: Callable[[Any], float] | None = None,
) -> Experiment | None:
    """The one act worth performing next, or None if none of them tells her anything.

    ``predicts(hypothesis, act)`` says what that account expects; returning
    None, or raising, means it will not say.

    Where two acts settle the same amount for the same cost the first offered
    wins, so a caller who hands over a settled order gets a settled answer —
    a set with nothing to choose between its members must not choose by
    whichever address happened to hash first.
    """
    if len(hypotheses) < 2:
        return None
    before = _how_likely_before_any_of_it(hypotheses, plausibility)
    best: Experiment | None = None
    for act in could_try:
        price = 1.0 if costs is None else max(0.0, float(costs(act)))
        weighed = _weigh_one_act(hypotheses, act, predicts, before, price)
        if weighed.settles <= 0:
            continue
        if best is None or weighed.worth_doing > best.worth_doing:
            best = weighed
    return best


def what_it_ruled_out(
    hypotheses: Mapping[str, Any],
    tried: Any,
    saw: Any,
    *,
    predicts: Callable[[Any, Any], Any],
) -> dict[str, Any]:
    """The accounts still standing after performing an act and seeing an outcome.

    An account that expected something else is gone. An account that would not
    say is untouched: it was never on the hook for this one.
    """
    said = _what_each_expects(hypotheses, tried, predicts)
    seen = _key(saw)
    return {
        name: one
        for name, one in hypotheses.items()
        if said[name] is SILENT or _key(said[name]) == seen
    }


@dataclass(frozen=True)
class WhatTheseActsSaw:
    """Which accounts these particular acts could not tell apart.

    Carrying the acts tried, because the answer means nothing without them: a
    thin act space makes different accounts look like one, and a caller reading
    ``groups`` alone would never know how hard she had actually looked.
    """

    #: Groups whose members agreed on every act offered.
    groups: tuple[tuple[str, ...], ...]
    #: How many acts that judgement rests on.
    acts_tried: int

    def __bool__(self) -> bool:
        return bool(self.groups)

    def __str__(self) -> str:
        if not self.groups:
            return f"{self.acts_tried} acts told all of them apart"
        return (
            f"{len(self.groups)} groups agreed on all {self.acts_tried} acts "
            "offered — which may mean they are one account, or may mean the "
            "acts offered were too few"
        )


def what_these_acts_cannot_separate(
    hypotheses: Mapping[str, Any],
    could_try: Iterable[Any],
    *,
    predicts: Callable[[Any, Any], Any],
) -> WhatTheseActsSaw:
    """Groups of accounts that agree on every act offered.

    Two accounts in one group may be one account wearing two names, in which
    case the shorter spelling is the one to keep. They may equally be two
    accounts and a set of acts too thin to part them. Nothing here can tell
    those cases apart, so nothing here pretends to: the answer says what it
    rests on and the caller decides whether that was enough.
    """
    acts = list(could_try)
    signatures: dict[tuple[Any, ...], list[str]] = {}
    for name, one in hypotheses.items():
        marks = []
        for act in acts:
            try:
                answer = predicts(one, act)
            except Exception:  # noqa: BLE001 - refusing to say IS an answer here
                answer = SILENT
            marks.append(_key(SILENT if answer is None else answer))
        signatures.setdefault(tuple(marks), []).append(name)
    return WhatTheseActsSaw(
        groups=tuple(
            tuple(sorted(names)) for names in signatures.values() if len(names) > 1
        ),
        acts_tried=len(acts),
    )


def how_many_would_settle_it(
    hypotheses: Mapping[str, Any],
    could_try: Iterable[Any],
    *,
    predicts: Callable[[Any, Any], Any],
    plausibility: Callable[[str, Any], float] | None = None,
) -> int | None:
    """Acts needed in the worst case, or None if these acts never settle it.

    She chooses greedily at each step, as she would in life, and the count is
    the deepest branch that choosing can lead to. None means some pair agrees
    on every act she was offered — which is a fact about the offer as much as
    about the pair.
    """
    acts = list(could_try)
    if what_these_acts_cannot_separate(hypotheses, acts, predicts=predicts):
        return None
    return _depth(dict(hypotheses), acts, predicts, plausibility, set())


def _depth(
    standing: dict[str, Any],
    acts: Sequence[Any],
    predicts: Callable[[Any, Any], Any],
    plausibility: Callable[[str, Any], float] | None,
    done: set[int],
) -> int:
    if len(standing) < 2:
        return 0
    left = [act for index, act in enumerate(acts) if index not in done]
    chosen = what_to_try(
        standing, left, predicts=predicts, plausibility=plausibility
    )
    if chosen is None:
        return 0
    where = next(
        index
        for index, act in enumerate(acts)
        if index not in done and act is chosen.do
    )
    deepest = 0
    for answer in {_key(one) for one in chosen.expects.values()}:
        after = {
            name: one
            for name, one in standing.items()
            if name in chosen.silent or _key(chosen.expects[name]) == answer
        }
        if len(after) == len(standing):
            continue
        deepest = max(
            deepest, _depth(after, acts, predicts, plausibility, done | {where})
        )
    return 1 + deepest


def every_act_that_settles_a_sequence(of_length: int = 4) -> tuple[tuple[int, ...], ...]:
    """An act space for sequences that leaves nothing out.

    The same enumeration the pairwise form walked, offered as acts so that the
    general chooser inherits its completeness rather than approximating it:
    all-distinct values separate any two readings that take from different
    places, and every state over two values separates any two doings that treat
    a pair differently, across the lengths where a reading can differ at all.
    """
    from core.cognition.an_invented_kind import (
        _LENGTHS_TO_SETTLE_IT,
        _every_telling_state,
    )

    acts: list[tuple[int, ...]] = []
    for size in sorted({int(of_length), *_LENGTHS_TO_SETTLE_IT}):
        if size >= 2:
            acts.extend(_every_telling_state(size))
    return tuple(acts)
