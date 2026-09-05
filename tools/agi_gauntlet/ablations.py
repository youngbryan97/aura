"""tools/agi_gauntlet/ablations.py — which part of her did the work.

Suppose the whole system passes. Then the system is what passed, and that is
a real result whatever the model underneath contributed. But the claim people
usually want is narrower — that the architecture caused the generality — and
that one needs the comparison:

    Aura(model)     against     a plain scaffold(model)

with the same model, the same quantisation, the same tools, the same token
and compute budget, the same environment and the same task information. Any
difference in those and the comparison is about the difference.

And then the parts, one at a time:

    Aura −memory   −development   −workspace   −self-model   −world-model

Each is a run of the same gates with one thing switched off. What a lesion is
worth is the gap it leaves, and a part whose removal changes nothing was not
doing the work whatever its docstring says.

What is here and what is not
----------------------------
The lesions that can be applied without a model run here: development off,
memory reset between episodes, and the world model taken away. The comparison
against a plain scaffold on the same weights cannot — it needs the weights,
and running a 32B beside a resident one on this host is how the machine dies.
It is declared, with what it needs, rather than approximated by something
cheaper wearing its name.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Lesion",
    "THE_LESIONS",
    "what_each_part_is_worth",
    "what_each_part_is_worth_in_company",
]


@dataclass(frozen=True)
class Lesion:
    """One thing switched off, and how to switch it off."""

    name: str
    what_it_removes: str
    #: None where this harness cannot apply it.
    apply: Callable[[], Any] | None = None
    needs: str = ""

    @property
    def can_be_applied(self) -> bool:
        return self.apply is not None


@contextmanager
def _without_development() -> Iterator[None]:
    """Her developmental policy sees nothing, so it can change nothing."""

    import core.cognition.the_record_of_her_own_work as record

    was = list(record.the_record().kept)
    record.forget_the_record()
    try:
        yield
    finally:
        record.the_record().kept[:] = was


@contextmanager
def _without_what_mattered() -> Iterator[None]:
    """What each run taught, thrown away before the next one.

    Not a clear before the gate starts — the gate clears it per world
    already, so a lesion applied outside removed nothing and reported a gap
    of zero for a part that is doing the work. It has to hold for the whole
    run, which is what the switch is for.
    """

    from core.agency.what_matters_here import forget_what_mattered, keep_nothing

    keep_nothing(True)
    try:
        yield
    finally:
        keep_nothing(False)
        forget_what_mattered()


@contextmanager
def _without_the_library() -> Iterator[None]:
    """The structures earlier worlds taught her, taken away."""

    from core.cognition.relation_language import RelationLanguage

    blank = RelationLanguage()
    try:
        yield blank
    finally:
        pass


@contextmanager
def _with_the_standing_guess() -> Iterator[None]:
    """Enter an unfamiliar world judging it by what mattered in the last one.

    The other half of `what_matters_here`, and the half that turns out to do
    the work: not the weights it learns, but its refusal to use a guess about
    what matters until there is an outcome to learn from.
    """

    from core.agency.what_matters_here import always_the_guess

    always_the_guess(True)
    try:
        yield
    finally:
        always_the_guess(False)


@contextmanager
def _without_newness() -> Iterator[None]:
    """The one term that separates going somewhere from pacing, taken away."""

    from core.agency import how_good_is_this as judging

    was = dict(judging.AS_GOOD_A_GUESS_AS_ANY)
    judging.AS_GOOD_A_GUESS_AS_ANY["newness"] = 0.0
    try:
        yield
    finally:
        judging.AS_GOOD_A_GUESS_AS_ANY.clear()
        judging.AS_GOOD_A_GUESS_AS_ANY.update(was)


THE_LESIONS: tuple[Lesion, ...] = (
    Lesion(
        "no newness",
        "the term that separates somewhere she has been from somewhere she has not",
        apply=_without_newness,
    ),
    Lesion(
        "no development",
        "the record her developmental policy reads",
        apply=_without_development,
    ),
    Lesion(
        "reset between episodes",
        "what she worked out about this world, between every episode",
        apply=_without_what_mattered,
    ),
    Lesion(
        "the standing guess from the first move",
        "the refusal to guess what matters before there is an outcome",
        apply=_with_the_standing_guess,
    ),
    Lesion(
        "no library",
        "the structures earlier worlds taught her",
        apply=_without_the_library,
    ),
    Lesion(
        "the model in a plain scaffold",
        "everything except the weights: a read-decide-act loop and nothing else",
        needs=(
            "the same weights, quantisation, tools, token budget, compute "
            "budget, environment and task information as the full system. "
            "Running a second 32B beside the resident one on this host is how "
            "the machine dies, so this runs where the weights can be loaded "
            "alone."
        ),
    ),
    Lesion(
        "no self-model",
        "what she believes about her own faculties",
        needs=(
            "a live runtime. The self-model is assembled at boot from the "
            "service container, and a lesion applied to a process that never "
            "booted removes nothing."
        ),
    ),
)


def what_each_part_is_worth(
    gate: Callable[[], dict[str, Any]], *, lesions: tuple[Lesion, ...] = THE_LESIONS
) -> dict[str, Any]:
    """Run one gate whole, then once per lesion, and report the gaps.

    A part whose removal changes nothing was not doing the work, and that is
    a finding rather than a disappointment: it is the only way to tell a
    component that matters from one that is present.
    """

    whole = gate()
    found: dict[str, Any] = {"whole": whole, "lesions": {}, "declared": []}
    for lesion in lesions:
        if not lesion.can_be_applied:
            found["declared"].append(
                {
                    "lesion": lesion.name,
                    "removes": lesion.what_it_removes,
                    "needs": lesion.needs,
                }
            )
            continue
        with lesion.apply():  # type: ignore[misc]
            hurt = gate()
        found["lesions"][lesion.name] = {
            "removes": lesion.what_it_removes,
            "result": hurt,
            "gap": _gap(whole, hurt),
        }
    return found


def what_each_part_is_worth_in_company(
    channels: Sequence[str],
    measure: Callable[[], float],
    *,
    permutations: int = 0,
) -> dict[str, Any]:
    """The same question, asked of every part at once instead of one at a time.

    One-at-a-time is wrong in a way that shows up here every run. A part with a
    duplicate costs nothing to remove — the twin covers — so it reads as doing
    nothing, and so does its twin, while the pair is essential. Two parts that
    only work together read as doing nothing too, for the opposite reason. The
    reading above says "a gap of zero is the only way to tell a component that
    matters from one that is present", and a gap of zero is exactly what both
    of those cases produce.

    ``core.verify.coalition_credit`` already solves this — a marginal
    contribution averaged over many backgrounds, with the interaction term
    saying which of the two cases a zero gap is — and nothing outside its own
    test had ever called it. It lesions through the real registry, so this
    measures the organism rather than a model of it.
    """

    from core.verify.coalition_credit import DEFAULT_PERMUTATIONS, attribute_registered

    found = attribute_registered(
        list(channels),
        measure,
        permutations=permutations or DEFAULT_PERMUTATIONS,
    )
    return {
        "trials": found.trials,
        "credits": [
            {
                "channel": one.channel,
                "leave_one_out": round(one.leave_one_out, 4),
                "marginal": round(one.marginal, 4),
                "interaction": round(one.interaction, 4),
                "role": str(one.role),
                "error": round(one.standard_error, 4),
            }
            for one in found.credits
        ],
    }


def _gap(whole: dict[str, Any], hurt: dict[str, Any]) -> dict[str, Any]:
    """What the lesion cost, on every number both runs reported."""

    return {
        key: round(float(whole[key]) - float(hurt[key]), 4)
        for key in whole
        if isinstance(whole.get(key), (int, float))
        and isinstance(hurt.get(key), (int, float))
        and not isinstance(whole.get(key), bool)
    }
