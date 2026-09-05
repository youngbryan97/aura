"""Soar's preference semantics: the part of a decision that a number cannot express.

What Aura had, and what was wrong with it
-----------------------------------------
Every candidate selection in this codebase reduces to a float. ``rank_actions``
scores actions with a value model, subtracts ``0.20 * risk``, and commits to the
argmax. :mod:`core.cognition.impasse` can then say the decision deadlocked, but
nothing anywhere can say *this option must not be taken* in a way the arithmetic
cannot overturn.

That is a real defect and not a stylistic one. A standing directive is a
prohibition the owner wrote down. Expressed as a risk of ``1.0`` it costs a
candidate ``0.20`` of value, so an action the owner forbade outright still wins
against an alternative worth ``0.25`` less. The refusal happens later, at the
gateway, which means deliberation spends its budget planning something it is not
allowed to do and the receipt records a commitment to a forbidden act.

Soar solved this in the early eighties by refusing to make preferences
commensurable. Symbolic preferences are resolved first, in a fixed order, and
only what survives them is eligible at all. Numbers enter last, and only among
candidates that something has explicitly declared interchangeable.

The order
---------
:func:`resolve` runs the eight steps of ``run_preference_semantics`` in Soar's
order, and the order carries the meaning:

1. **require** — one candidate is mandatory. Two are a contradiction, and a
   mandatory candidate that is also prohibited is a contradiction, so both raise
   :attr:`~core.cognition.impasse.ImpasseType.CONSTRAINT_FAILURE`.
2. **acceptable** — nothing is a candidate until something proposes it.
3. **prohibit** and **reject** — removal. Prohibit is permanent and reject is
   for this decision; neither is a quantity.
4. **better/worse** — pairwise dominance. If dominance eliminates everything,
   the preferences contradict each other and that is a
   :attr:`~core.cognition.impasse.ImpasseType.CONFLICT`.
5. **best** — narrow to the ones marked best, if any are.
6. **worst** — drop the ones marked worst, unless that would empty the field.
7. **indifferent** — the survivors are interchangeable only if something says
   so. Otherwise the decision procedure has genuinely failed to choose and the
   result is a :attr:`~core.cognition.impasse.ImpasseType.TIE`.

Why require beats reject but not prohibit
-----------------------------------------
Soar terminates at step 1 with the required candidate, so a require silently
overrides a reject. That looks like a hole until you read the two as English:
reject means "not this time" and prohibit means "never". A requirement can
outrank a scheduling objection; it cannot outrank a ban, which is why the
prohibit check is inside step 1 and the reject check is not. This
implementation keeps that asymmetry deliberately rather than tidying it away.

Why indifference has to be asserted
-----------------------------------
The interesting half of step 7 is what happens when nobody asserts it. Two
candidates that no preference distinguishes are *not* interchangeable — they are
undecided, and picking one is arbitrary. Sorting and taking ``[0]`` makes that
arbitrary choice look like a decision and destroys the evidence that it was not.
So an unasserted tie becomes an impasse, which
:class:`~core.cognition.impasse.ImpasseLearner` already knows how to count and
learn from, and only an *asserted* indifference reaches the selection policy.

Determinism
-----------
When candidates are declared indifferent, something still has to choose.
:class:`SituationSeededUniform` seeds a generator from the situation signature,
so the same deadlock in the same situation resolves the same way on every run
and on every replay — reproducible without being an artifact of the order the
candidates arrived in. :class:`BoltzmannSelection` is the exploring alternative
for numeric-indifferent values, which is where Soar-RL attaches.

This module decides nothing on its own and touches no global state. It takes
candidates and preferences and returns a :class:`Resolution` that names every
removal and the preference that caused it.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from core.cognition.impasse import Impasse, ImpasseType, situation_signature

__all__ = [
    "PreferenceType",
    "Preference",
    "PreferenceSet",
    "ResolutionStep",
    "Resolution",
    "IndifferentSelection",
    "SituationSeededUniform",
    "DeclaredValueGreedy",
    "BoltzmannSelection",
    "PreferenceBuilder",
    "resolve",
    "acceptable",
    "reject",
    "require",
    "prohibit",
    "better",
    "worse",
    "best",
    "worst",
    "indifferent",
    "binary_indifferent",
    "numeric_indifferent",
]


class PreferenceType(StrEnum):
    """The preference vocabulary, split by arity.

    Unary preferences carry an ``item``. Binary preferences carry an ``item``
    and a ``reference`` and are read left to right: ``BETTER`` means *item is
    better than reference*. ``NUMERIC_INDIFFERENT`` is the one unary preference
    that also carries a ``value``, and it is the seam where learned action
    values enter a decision the symbolic layer has already declared open.
    """

    ACCEPTABLE = "acceptable"
    REJECT = "reject"
    REQUIRE = "require"
    PROHIBIT = "prohibit"
    BEST = "best"
    WORST = "worst"
    BETTER = "better"
    WORSE = "worse"
    INDIFFERENT = "indifferent"
    BINARY_INDIFFERENT = "binary_indifferent"
    NUMERIC_INDIFFERENT = "numeric_indifferent"


_BINARY_TYPES = frozenset(
    {PreferenceType.BETTER, PreferenceType.WORSE, PreferenceType.BINARY_INDIFFERENT}
)


@dataclass(frozen=True)
class Preference:
    """One assertion about a candidate, and who made it.

    ``source`` is not decoration. A resolution that removes a candidate has to
    be able to say which subsystem removed it, or the refusal downstream cannot
    be explained to the person it affects. Every constructor in this module
    requires one.
    """

    type: PreferenceType
    item: str
    reference: str = ""
    value: float = 0.0
    source: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.item:
            raise ValueError("a preference must name an item")
        if not self.source:
            raise ValueError(f"{self.type.value} preference for {self.item!r} has no source")
        if self.type in _BINARY_TYPES and not self.reference:
            raise ValueError(f"{self.type.value} is binary and needs a reference item")
        if self.type not in _BINARY_TYPES and self.reference:
            raise ValueError(f"{self.type.value} is unary and cannot take a reference item")
        if self.type is PreferenceType.NUMERIC_INDIFFERENT and not math.isfinite(self.value):
            raise ValueError("numeric-indifferent value must be finite")

    def __str__(self) -> str:
        if self.type in _BINARY_TYPES:
            body = f"{self.item} {self.type.value} {self.reference}"
        elif self.type is PreferenceType.NUMERIC_INDIFFERENT:
            body = f"{self.item} = {self.value:g}"
        else:
            body = f"{self.item} {self.type.value}"
        return f"{body} [{self.source}]"


def _make(kind: PreferenceType):
    def build(item: str, *, source: str, detail: str = "") -> Preference:
        return Preference(type=kind, item=item, source=source, detail=detail)

    build.__name__ = kind.value
    build.__doc__ = f"A unary {kind.value} preference for ``item``."
    return build


acceptable = _make(PreferenceType.ACCEPTABLE)
reject = _make(PreferenceType.REJECT)
require = _make(PreferenceType.REQUIRE)
prohibit = _make(PreferenceType.PROHIBIT)
best = _make(PreferenceType.BEST)
worst = _make(PreferenceType.WORST)
indifferent = _make(PreferenceType.INDIFFERENT)


def better(item: str, than: str, *, source: str, detail: str = "") -> Preference:
    """``item`` dominates ``than``."""
    return Preference(
        type=PreferenceType.BETTER, item=item, reference=than, source=source, detail=detail
    )


def worse(item: str, than: str, *, source: str, detail: str = "") -> Preference:
    """``item`` is dominated by ``than``."""
    return Preference(
        type=PreferenceType.WORSE, item=item, reference=than, source=source, detail=detail
    )


def binary_indifferent(item: str, other: str, *, source: str, detail: str = "") -> Preference:
    """Choosing between these two by any means is acceptable."""
    return Preference(
        type=PreferenceType.BINARY_INDIFFERENT,
        item=item,
        reference=other,
        source=source,
        detail=detail,
    )


def numeric_indifferent(
    item: str, value: float, *, source: str, detail: str = ""
) -> Preference:
    """Interchangeable with other numeric-indifferent candidates, carrying a value.

    The value biases the selection policy and nothing else. It cannot rescue a
    prohibited candidate or break a tie the symbolic layer left open, which is
    the whole reason learned values are admitted here and not earlier.
    """
    return Preference(
        type=PreferenceType.NUMERIC_INDIFFERENT,
        item=item,
        value=float(value),
        source=source,
        detail=detail,
    )


class PreferenceSet:
    """Indexed preferences over one decision.

    Built once per decision and read many times by :func:`resolve`. Preferences
    naming an item that is not a candidate are kept but never consulted, which
    matches Soar: a preference for an operator nobody proposed is inert rather
    than an error.
    """

    __slots__ = ("_all", "_unary", "_binary")

    def __init__(self, preferences: Iterable[Preference] = ()) -> None:
        self._all: list[Preference] = []
        self._unary: dict[PreferenceType, dict[str, list[Preference]]] = {}
        self._binary: dict[PreferenceType, dict[tuple[str, str], list[Preference]]] = {}
        for pref in preferences:
            self.add(pref)

    def add(self, pref: Preference) -> None:
        if not isinstance(pref, Preference):
            raise TypeError(f"expected Preference, got {type(pref).__name__}")
        self._all.append(pref)
        if pref.type in _BINARY_TYPES:
            table = self._binary.setdefault(pref.type, {})
            table.setdefault((pref.item, pref.reference), []).append(pref)
        else:
            table = self._unary.setdefault(pref.type, {})
            table.setdefault(pref.item, []).append(pref)

    def __len__(self) -> int:
        return len(self._all)

    def __iter__(self):
        return iter(self._all)

    def unary(self, kind: PreferenceType, item: str) -> list[Preference]:
        return list(self._unary.get(kind, {}).get(item, ()))

    def has(self, kind: PreferenceType, item: str) -> bool:
        return bool(self._unary.get(kind, {}).get(item))

    def items_with(self, kind: PreferenceType, among: Sequence[str]) -> list[str]:
        table = self._unary.get(kind, {})
        return [c for c in among if table.get(c)]

    def binary(self, kind: PreferenceType) -> list[Preference]:
        table = self._binary.get(kind, {})
        return [p for prefs in table.values() for p in prefs]

    def binary_between(self, kind: PreferenceType, a: str, b: str) -> bool:
        table = self._binary.get(kind, {})
        return bool(table.get((a, b)) or table.get((b, a)))

    def numeric_value(self, item: str) -> float | None:
        """The declared numeric-indifferent value, or None.

        Several assertions for one item are averaged rather than last-wins: two
        subsystems that both measured this action are two observations, and
        discarding one because it was added first would make the result depend
        on iteration order — the defect this module exists to remove.
        """
        prefs = self._unary.get(PreferenceType.NUMERIC_INDIFFERENT, {}).get(item)
        if not prefs:
            return None
        return sum(p.value for p in prefs) / len(prefs)


@dataclass(frozen=True)
class ResolutionStep:
    """One stage of the procedure, and what it did.

    Kept for every stage, including the ones that changed nothing, because
    "step 4 removed nothing" and "step 4 never ran" are different explanations
    and an audit trail that cannot tell them apart is not one.
    """

    stage: str
    before: tuple[str, ...]
    after: tuple[str, ...]
    removed: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return self.before != self.after


@dataclass(frozen=True)
class Resolution:
    """What the decision procedure produced.

    Exactly one of ``winner`` and ``impasse`` is set. ``survivors`` is what
    remained when the procedure stopped, which is the candidate set a substate
    should deliberate over — on a tie that is the tied set, not the original
    field, so the substate inherits the narrowing the symbolic layer achieved.
    """

    winner: str | None
    impasse: Impasse | None
    survivors: tuple[str, ...]
    steps: tuple[ResolutionStep, ...] = ()
    selection_reason: str = ""
    signature: str = ""

    def __post_init__(self) -> None:
        if (self.winner is None) == (self.impasse is None):
            raise ValueError("a resolution is exactly one of a winner or an impasse")

    @property
    def decided(self) -> bool:
        return self.winner is not None

    def why(self, candidate: str) -> str:
        """The stage and reason that removed ``candidate``, or why it survived."""
        for step in self.steps:
            if candidate in step.removed:
                idx = step.removed.index(candidate)
                reason = step.reasons[idx] if idx < len(step.reasons) else step.stage
                return f"{step.stage}: {reason}"
        if candidate in self.survivors:
            return "survived every stage"
        return "was never a candidate"

    def to_dict(self) -> dict[str, object]:
        return {
            "winner": self.winner,
            "impasse": None if self.impasse is None else self.impasse.type.value,
            "impasse_detail": "" if self.impasse is None else self.impasse.detail,
            "survivors": list(self.survivors),
            "selection_reason": self.selection_reason,
            "signature": self.signature,
            "steps": [
                {
                    "stage": s.stage,
                    "removed": list(s.removed),
                    "reasons": list(s.reasons),
                    "remaining": len(s.after),
                }
                for s in self.steps
                if s.changed
            ],
        }


@runtime_checkable
class IndifferentSelection(Protocol):
    """How to choose once the symbolic layer has declared the field open."""

    name: str

    def choose(self, candidates: Sequence[str], values: Mapping[str, float]) -> str:
        """Pick one. ``candidates`` is non-empty and sorted."""
        ...


@dataclass(frozen=True)
class SituationSeededUniform:
    """Uniform over the tied set, seeded by the situation.

    A live cognitive loop that flaps between two indifferent options on
    identical inputs cannot be debugged and cannot be replayed. Seeding from the
    candidate set gives a choice that is stable for a given situation and still
    independent of the order the candidates were submitted in, which is the
    property that was actually missing. Two different situations that happen to
    offer the same options get different draws, so this is not a disguised fixed
    ordering.

    Declared values are ignored here by design. This is the policy for a set
    where nothing measured anything, and :class:`DeclaredValueGreedy` is what
    picks it when that is the case.
    """

    name: str = "situation_seeded_uniform"

    def choose(self, candidates: Sequence[str], values: Mapping[str, float]) -> str:
        del values
        return candidates[0] if len(candidates) == 1 else _seeded(candidates).choice(list(candidates))


@dataclass(frozen=True)
class DeclaredValueGreedy:
    """Take the highest declared value; draw uniformly when the evidence is uneven.

    Two failure modes are being avoided at once.

    Discarding the values is one. A caller that measured these actions and said
    so has produced the only real information in the decision, and a uniform
    draw over the set throws it away.

    Ranking on a *partial* declaration is the other, and it is worse. If three
    candidates are open and only two carry values, preferring a measured
    candidate over an unmeasured one is not a judgement about the actions — it
    is a judgement about which ones somebody happened to have data for. That is
    the same mistake ``core/reasoning/action_value.py`` was rewritten to stop
    making, where an unevidenced midpoint could be ranked against a measurement
    as though the two were comparable. So a partial declaration falls back to
    the uniform draw and the reason is reported.

    Equal top values fall back to the seeded draw as well, which is what keeps
    the policy total: something always gets chosen, and never by list position.
    """

    name: str = "declared_value_greedy"

    def choose(self, candidates: Sequence[str], values: Mapping[str, float]) -> str:
        if len(candidates) == 1:
            return candidates[0]
        if len(values) != len(candidates) or not values:
            return _seeded(candidates).choice(list(candidates))
        top = max(values[c] for c in candidates)
        leaders = [c for c in candidates if values[c] >= top]
        if len(leaders) == 1:
            return leaders[0]
        return _seeded(leaders).choice(leaders)


@dataclass(frozen=True)
class BoltzmannSelection:
    """Softmax over declared numeric-indifferent values — the Soar-RL seam.

    ``temperature`` has no default. Any value here is a claim about how much
    exploration this decision warrants, and a module-level constant would be
    that claim made once, invisibly, for every caller.

    Candidates with no declared value do not silently score zero, which would
    read as "average" on a scale that may not contain zero. They are given the
    mean of the declared values, so an undeclared candidate is drawn as often as
    a typical declared one and the absence of a measurement never masquerades
    as a measurement.
    """

    temperature: float
    name: str = "boltzmann"

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("Boltzmann temperature must be finite and positive")

    def choose(self, candidates: Sequence[str], values: Mapping[str, float]) -> str:
        if len(candidates) == 1:
            return candidates[0]
        declared = [values[c] for c in candidates if c in values]
        fallback = sum(declared) / len(declared) if declared else 0.0
        scores = [values.get(c, fallback) for c in candidates]
        # Shift by the maximum before exponentiating. Without it a value of a
        # few hundred overflows to inf and the softmax returns nan.
        top = max(scores)
        weights = [math.exp((s - top) / self.temperature) for s in scores]
        total = sum(weights)
        if not math.isfinite(total) or total <= 0.0:
            return _seeded(candidates).choice(list(candidates))
        return _seeded(candidates).choices(list(candidates), weights=weights, k=1)[0]


def _seeded(candidates: Sequence[str]) -> random.Random:
    """A generator seeded by the candidate set, order-independently.

    The candidates are sorted before hashing so that submitting ``[a, b]`` and
    ``[b, a]`` seeds identically. Using the process-wide ``random`` module
    instead would make every arbitrary choice depend on how many other draws
    happened first, which is the same unreproducibility in a new place.
    """
    digest = hashlib.blake2b(
        # NUL separates, because it cannot occur inside a candidate name.
        # With a space, ["a b", "c"] and ["a", "b c"] hash identically and
        # two different decisions would share a draw.
        "\x00".join(sorted(candidates)).encode("utf-8"), digest_size=8
    ).digest()
    return random.Random(int.from_bytes(digest, "big"))


#: Greedy on declared values, uniform when there are none or only some. Chosen
#: as the default because it is the policy that adds no information of its own:
#: it uses a measurement where one exists and refuses to invent an ordering
#: where one does not. Exploration is a real and separate decision, so
#: :class:`BoltzmannSelection` has to be asked for.
_DEFAULT_SELECTION = DeclaredValueGreedy()


def resolve(
    candidates: Sequence[str],
    preferences: Iterable[Preference] | PreferenceSet = (),
    *,
    selection: IndifferentSelection | None = None,
    context: Mapping[str, object] | None = None,
) -> Resolution:
    """Run the eight-step procedure over ``candidates``.

    Returns a :class:`Resolution` holding either the winner or the typed
    impasse, plus the per-stage trace. Duplicate candidates are collapsed and
    the original submission order is preserved for everything except the seeded
    draw, so a caller can still read the trace in the order it thinks in.
    """
    prefs = preferences if isinstance(preferences, PreferenceSet) else PreferenceSet(preferences)
    ordered: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        text = str(c)
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)

    ctx = dict(context or {})
    signature = situation_signature(ctx, ordered)
    steps: list[ResolutionStep] = []

    def record(stage: str, before: Sequence[str], after: Sequence[str], reasons: Mapping[str, str]) -> None:
        removed = tuple(c for c in before if c not in set(after))
        steps.append(
            ResolutionStep(
                stage=stage,
                before=tuple(before),
                after=tuple(after),
                removed=removed,
                reasons=tuple(reasons.get(c, stage) for c in removed),
            )
        )

    def impasse_of(kind: ImpasseType, field_: Sequence[str], detail: str) -> Resolution:
        return Resolution(
            winner=None,
            impasse=Impasse(
                type=kind,
                signature=situation_signature(ctx, field_) if field_ else signature,
                candidates=tuple(sorted(field_)),
                detail=detail,
            ),
            survivors=tuple(field_),
            steps=tuple(steps),
            signature=signature,
        )

    # -- 1. require ------------------------------------------------------
    required = prefs.items_with(PreferenceType.REQUIRE, ordered)
    if len(required) > 1:
        sources = ", ".join(
            f"{c} ({'; '.join(p.source for p in prefs.unary(PreferenceType.REQUIRE, c))})"
            for c in required
        )
        record("require", ordered, ordered, {})
        return impasse_of(
            ImpasseType.CONSTRAINT_FAILURE,
            required,
            f"{len(required)} candidates are each required: {sources}",
        )
    if len(required) == 1:
        winner = required[0]
        banned = prefs.unary(PreferenceType.PROHIBIT, winner)
        if banned:
            record("require", ordered, ordered, {})
            return impasse_of(
                ImpasseType.CONSTRAINT_FAILURE,
                [winner],
                f"{winner} is required and prohibited "
                f"({'; '.join(p.source for p in banned)})",
            )
        record("require", ordered, [winner], {c: "not required" for c in ordered if c != winner})
        return Resolution(
            winner=winner,
            impasse=None,
            survivors=(winner,),
            steps=tuple(steps),
            selection_reason="required by "
            + "; ".join(p.source for p in prefs.unary(PreferenceType.REQUIRE, winner)),
            signature=signature,
        )
    record("require", ordered, ordered, {})

    # -- 2. acceptable ---------------------------------------------------
    live = prefs.items_with(PreferenceType.ACCEPTABLE, ordered)
    record("acceptable", ordered, live, {c: "never proposed as acceptable" for c in ordered})
    if not live:
        return impasse_of(
            ImpasseType.REJECTION,
            (),
            "no candidates were proposed" if not ordered else "no candidate was acceptable",
        )

    # -- 3. prohibit and reject ------------------------------------------
    reasons: dict[str, str] = {}
    kept: list[str] = []
    for c in live:
        blocking = prefs.unary(PreferenceType.PROHIBIT, c) or prefs.unary(PreferenceType.REJECT, c)
        if blocking:
            kind = blocking[0].type.value
            detail = "; ".join(p.detail or p.source for p in blocking)
            reasons[c] = f"{kind} by {detail}"
        else:
            kept.append(c)
    record("prohibit/reject", live, kept, reasons)
    if not kept:
        return impasse_of(
            ImpasseType.REJECTION,
            (),
            "every candidate was prohibited or rejected: "
            + "; ".join(f"{c} — {reasons[c]}" for c in live),
        )
    live = kept

    # -- 4. better/worse -------------------------------------------------
    # One pass, exactly as Soar does it: a candidate that anything dominates is
    # out, with no transitive closure. That is what makes a > b, b > a remove
    # both and surface as a conflict rather than resolving to whichever
    # preference was asserted first.
    live_set = set(live)
    dominated: dict[str, str] = {}
    for pref in prefs.binary(PreferenceType.BETTER):
        if pref.item in live_set and pref.reference in live_set:
            dominated.setdefault(pref.reference, f"worse than {pref.item} per {pref.source}")
    for pref in prefs.binary(PreferenceType.WORSE):
        if pref.item in live_set and pref.reference in live_set:
            dominated.setdefault(pref.item, f"worse than {pref.reference} per {pref.source}")
    undominated = [c for c in live if c not in dominated]
    record("better/worse", live, undominated, dominated)
    if not undominated:
        return impasse_of(
            ImpasseType.CONFLICT,
            live,
            "dominance preferences eliminated every candidate: "
            + "; ".join(f"{c} — {dominated[c]}" for c in live),
        )
    live = undominated

    # -- 5. best ---------------------------------------------------------
    marked_best = prefs.items_with(PreferenceType.BEST, live)
    if marked_best:
        record("best", live, marked_best, {c: "another candidate is marked best" for c in live})
        live = marked_best
    else:
        record("best", live, live, {})

    # -- 6. worst --------------------------------------------------------
    marked_worst = set(prefs.items_with(PreferenceType.WORST, live))
    non_worst = [c for c in live if c not in marked_worst]
    if marked_worst and non_worst:
        record("worst", live, non_worst, {c: "marked worst" for c in marked_worst})
        live = non_worst
    else:
        # Every survivor being marked worst is not a reason to have no
        # candidates. Soar keeps them, and so does this: worst is a relative
        # statement and it says nothing when there is nothing better left.
        record("worst", live, live, {})

    # -- 7. indifferent --------------------------------------------------
    if len(live) == 1:
        return Resolution(
            winner=live[0],
            impasse=None,
            survivors=tuple(live),
            steps=tuple(steps),
            selection_reason="sole survivor of preference resolution",
            signature=signature,
        )

    undeclared = _first_non_indifferent_pair(live, prefs)
    if undeclared is not None:
        a, b = undeclared
        return impasse_of(
            ImpasseType.TIE,
            live,
            f"{len(live)} candidates survive and nothing declares "
            f"{a} and {b} interchangeable",
        )

    policy = selection or _DEFAULT_SELECTION
    values = {c: v for c in live if (v := prefs.numeric_value(c)) is not None}
    winner = policy.choose(sorted(live), values)
    if winner not in live:
        raise ValueError(
            f"selection policy {policy.name!r} returned {winner!r}, which was not a candidate"
        )
    detail = f" over declared values {values}" if values else ""
    return Resolution(
        winner=winner,
        impasse=None,
        survivors=tuple(live),
        steps=tuple(steps),
        selection_reason=f"indifferent set resolved by {policy.name}{detail}",
        signature=signature,
    )


def _first_non_indifferent_pair(
    live: Sequence[str], prefs: PreferenceSet
) -> tuple[str, str] | None:
    """The first pair nothing declares interchangeable, or None.

    Returning the offending pair rather than a bare bool is what lets the tie
    impasse name what was undeclared. "Three candidates tied" tells an operator
    nothing; "nothing declares retry and escalate interchangeable" tells them
    precisely which preference is missing.
    """
    unary = {c for c in live if prefs.has(PreferenceType.INDIFFERENT, c)}
    numeric = {c for c in live if prefs.has(PreferenceType.NUMERIC_INDIFFERENT, c)}
    open_set = unary | numeric
    for i, a in enumerate(live):
        for b in live[i + 1 :]:
            if a in open_set and b in open_set:
                continue
            if prefs.binary_between(PreferenceType.BINARY_INDIFFERENT, a, b):
                continue
            return a, b
    return None


@dataclass
class PreferenceBuilder:
    """Accumulates preferences for one decision with a fixed source label.

    Callers that assert many preferences from one subsystem would otherwise
    repeat ``source=`` on every line, and a repeated argument is a place for one
    of them to drift. The builder makes the provenance a property of the
    subsystem doing the asserting, which is what it actually is.
    """

    source: str
    _prefs: list[Preference] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not str(self.source).strip():
            raise ValueError("a preference builder needs a source")

    def add(self, pref: Preference) -> PreferenceBuilder:
        self._prefs.append(pref)
        return self

    def acceptable(self, item: str, detail: str = "") -> PreferenceBuilder:
        return self.add(acceptable(item, source=self.source, detail=detail))

    def reject(self, item: str, detail: str = "") -> PreferenceBuilder:
        return self.add(reject(item, source=self.source, detail=detail))

    def require(self, item: str, detail: str = "") -> PreferenceBuilder:
        return self.add(require(item, source=self.source, detail=detail))

    def prohibit(self, item: str, detail: str = "") -> PreferenceBuilder:
        return self.add(prohibit(item, source=self.source, detail=detail))

    def best(self, item: str, detail: str = "") -> PreferenceBuilder:
        return self.add(best(item, source=self.source, detail=detail))

    def worst(self, item: str, detail: str = "") -> PreferenceBuilder:
        return self.add(worst(item, source=self.source, detail=detail))

    def better(self, item: str, than: str, detail: str = "") -> PreferenceBuilder:
        return self.add(better(item, than, source=self.source, detail=detail))

    def indifferent(self, item: str, detail: str = "") -> PreferenceBuilder:
        return self.add(indifferent(item, source=self.source, detail=detail))

    def numeric_indifferent(self, item: str, value: float, detail: str = "") -> PreferenceBuilder:
        return self.add(numeric_indifferent(item, value, source=self.source, detail=detail))

    def build(self) -> PreferenceSet:
        return PreferenceSet(self._prefs)

    def __iter__(self):
        return iter(self._prefs)

    def __len__(self) -> int:
        return len(self._prefs)
