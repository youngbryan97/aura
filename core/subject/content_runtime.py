"""Presenting percepts to the organism, and reading the two geometries out.

The percept classes are not invented here. The affect phase carries the table
that says which emotions each kind of percept may move, and that table is the
vocabulary percepts actually arrive in. One class per kind, at the intensity a
producer with no opinion writes. The design coordinate of a class is the set of
emotions its kind names, as a multi-hot vector over every emotion in the table,
so the designed distance between two classes is how differently the mechanism
is built to treat them. No number in the grid was chosen by whoever wrote the
experiment.

Each class is presented the way the world ordinarily arrives: through
`Condition.prepare`, on a common calm host reading, from a common fork. So two
classes differ in the percept and in nothing else.

Two readings come back from the same turns.

The state vector at the lag is the internal reading, and the Fisher-Rao
distance between two classes' state-vector samples is the internal geometry.

Which memories came back is the behavioural reading. It is a set of items
rather than a number, it is produced by retrieval rather than by the recorder,
and no column of the state vector carries it: the vector keeps how many came
back and how well the best one matched, never which ones. So the two geometries
share the percepts and share nothing else.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.state.percepts import PERCEPT_EMOTIONS, emit_percept
from core.subject.content import PerceptClass
from core.subject.intrinsic_v25 import crossfit_fisher_rao
from core.subject.perturbation import shift_reference
from core.subject.state import perturb, perturb_organs

__all__ = [
    "EMOTIONS",
    "KINDS",
    "ClassSamples",
    "behavioural_floor",
    "behavioural_geometry",
    "internal_geometry",
    "grid",
    "present",
    "sample_classes",
]

#: How strongly every class is presented: the top of the scale. At the default
#: half, no class was ever the most salient thing in front of her. The fork
#: still held her own percepts from the turn before, the turn at 0.83 and the
#: host's pressure near 0.78, and the most salient percept is the one recall is
#: cued by, so every class recalled what the turn cued and the behavioural
#: geometry was flat in every run. A stimulus the stream outranks has not been
#: presented. See tests/test_a_percept_class_is_presented_not_merely_emitted.py.
PRESENTED_INTENSITY: float = 1.0

#: The kind that is capped and rewritten before it reaches the table, so it is
#: not one percept class among others and is left out of the grid.
_SPECIAL: frozenset[str] = frozenset({"neural_decode"})

#: Every emotion any percept kind can move, in a fixed order. The coordinate
#: space of the design.
EMOTIONS: tuple[str, ...] = tuple(
    sorted({emotion for names in PERCEPT_EMOTIONS.values() for emotion in names})
)

#: The percept kinds the grid is built from: every kind the mechanism names,
#: less the one it special-cases.
KINDS: tuple[str, ...] = tuple(
    sorted(k for k, v in PERCEPT_EMOTIONS.items() if v and k not in _SPECIAL)
)


def grid() -> tuple[PerceptClass, ...]:
    """One class per percept kind, with the design read off the affect table."""
    out: list[PerceptClass] = []
    for kind in KINDS:
        named = set(PERCEPT_EMOTIONS[kind])
        coordinates = tuple(1.0 if emotion in named else 0.0 for emotion in EMOTIONS)
        out.append(
            PerceptClass(
                name=kind,
                kind=kind,
                source="world",
                intensity=PRESENTED_INTENSITY,
                # The content is the kind's own name. A percept carries text and
                # the text has to be something; making it the kind keeps the two
                # from varying independently and keeps the class one thing.
                content=kind.replace("_", " "),
                coordinates=coordinates,
            )
        )
    return tuple(out)


def present(cls: PerceptClass) -> Any:
    """A `prepare` that puts exactly this class in front of her.

    The host reading is the calm one every ordinary condition uses, so the body
    is in the same place for every class and the only difference is the percept.
    """
    from core.subject.driver import _calm

    def prepare(state: Any, rng: Any) -> dict[str, float]:
        reading = _calm(state, rng)
        emit_percept(
            state.world,
            cls.kind,
            content=cls.content,
            intensity=float(cls.intensity),
            source=cls.source,
        )
        reading["percept_arrived"] = 1.0
        return reading

    return prepare


@dataclass
class ClassSamples:
    """What one percept class produced, per anchor.

    Two arms, because the floor a distance is read against has to be the same
    comparison. Splitting one class's samples by anchor gave an unpaired
    distance — every anchor differs from every other far more than two percepts
    do — and the first run of this measured a same-class floor of 3.91 against
    between-class distances of 0.16. The second arm presents the same class from
    the same anchor again, so the floor is paired exactly as the comparison is.
    """

    name: str
    futures: list[np.ndarray]
    retrieved: list[frozenset[str]]
    context: list[np.ndarray]
    futures_b: list[np.ndarray] = field(default_factory=list)
    retrieved_b: list[frozenset[str]] = field(default_factory=list)


#: Recall writes each recollection with the score it was kept at, and that
#: score moves with her mood while the memory stays the same one. The
#: behavioural geometry is about which memories came back, so the score is taken
#: off before two recalls are compared; the store's own label stays.
_SCORE_MARK = re.compile(r"^\[memory score=[-0-9.]+\]\s*")


def _retrieved(runtime: Any) -> frozenset[str]:
    cognition = getattr(getattr(runtime, "state", None), "cognition", None)
    items = list(getattr(cognition, "long_term_memory", []) or [])
    return frozenset(_SCORE_MARK.sub("[memory] ", str(item)) for item in items)


async def sample_classes(
    runtime: Any,
    anchors: Sequence[Any],
    conditions: Sequence[Any],
    classes: Sequence[PerceptClass],
    *,
    turns: int = 1,
    lag: int = 1,
    displace: tuple[str, float] | None = None,
    reference: float | None = None,
) -> dict[str, ClassSamples]:
    """Present every class from every anchor, and read both geometries' inputs.

    `displace` moves one domain before the percept arrives. That is how the
    manifold is perturbed: the same classes are presented into a state that has
    been pushed, and both geometries are recomputed on what comes back.

    `reference` moves the point her feelings are measured from, towards feeling
    good by that much (`perturbation.shift_reference`). Unlike a push of the
    feelings it lasts the turn and leaves them free to answer the percept.
    """
    from dataclasses import replace

    out: dict[str, ClassSamples] = {
        cls.name: ClassSamples(cls.name, [], [], []) for cls in classes
    }

    async def one_arm(anchor: Any, condition: Any, cls: PerceptClass) -> tuple[Any, frozenset[str]] | None:
        runtime.restore(anchor.snapshot)
        if displace is not None:
            domain, delta = displace
            perturb(runtime.state, domain, delta, ontogeny=runtime.ontogeny)
            await perturb_organs(runtime.organs, domain, delta, state=runtime.state)
        if reference is not None:
            shift_reference(runtime.state, reference)
        shown = replace(condition, prepare=present(cls))
        rows: list[Any] = []
        for _ in range(turns):
            rows.extend(await runtime.turn_once(shown))
        if not rows:
            return None
        # The end of the `lag`-th turn after the fork. A turn records a frame
        # before its phases run and one after each, so reading frame `lag`
        # read the open frame: the percept was in the stream and no phase had
        # taken it in. Every class forked from one anchor had the same future
        # there, and the internal geometry was one distance for every pair.
        per_turn = max(1, len(rows) // max(1, turns))
        index = min(max(1, lag) * per_turn - 1, len(rows) - 1)
        return np.asarray(rows[index].vector(), dtype=np.float64), _retrieved(runtime)

    for anchor in anchors:
        for condition in conditions:
            for cls in classes:
                first = await one_arm(anchor, condition, cls)
                second = await one_arm(anchor, condition, cls)
                if first is None or second is None:
                    continue
                slot = out[cls.name]
                slot.futures.append(first[0])
                slot.retrieved.append(first[1])
                slot.futures_b.append(second[0])
                slot.retrieved_b.append(second[1])
                slot.context.append(np.asarray(anchor.current, dtype=np.float64))
    return out


def internal_geometry(
    samples: dict[str, ClassSamples],
    classes: Sequence[PerceptClass],
    *,
    folds: int = 5,
    seed: int = 2604,
    with_context: bool = True,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """Fisher-Rao distance between every pair of classes, and the same-class floor.

    The context is the anchor each sample forked from, appended to both arms.
    Its marginal is identical on both sides, so discrimination can only come
    from the conditional future, which is the thing being compared.
    """
    distances: dict[tuple[int, int], float] = {}
    floor: dict[tuple[int, int], float] = {}
    names = [cls.name for cls in classes]

    def block(name: str) -> np.ndarray | None:
        rows = samples.get(name)
        if rows is None or len(rows.futures) < folds:
            return None
        future = np.vstack(rows.futures)
        if not with_context or not rows.context:
            return future
        return np.hstack([np.vstack(rows.context), future])

    def block_b(name: str) -> np.ndarray | None:
        rows = samples.get(name)
        if rows is None or len(rows.futures_b) < folds:
            return None
        future = np.vstack(rows.futures_b)
        if not with_context or not rows.context:
            return future
        return np.hstack([np.vstack(rows.context)[: len(future)], future])

    blocks = {name: block(name) for name in names}
    for i, left_name in enumerate(names):
        left = blocks[left_name]
        if left is None:
            continue
        # The floor for this class: the same class presented again from the
        # same anchors, scored by the same estimator. Paired the way the
        # comparison is — splitting one arm by anchor instead measured how far
        # apart two anchors are, which is an order of magnitude larger than any
        # difference between percepts and made every distance look like noise.
        second = block_b(left_name)
        if second is not None and len(second) >= folds:
            same = crossfit_fisher_rao(
                left, second, folds=folds, seed=seed
            )
            floor[(i, i)] = float(same.distance_sq)
        for j in range(i + 1, len(names)):
            right = blocks[names[j]]
            if right is None:
                continue
            estimate = crossfit_fisher_rao(left, right, folds=folds, seed=seed)
            distances[(i, j)] = float(estimate.distance_sq)
    return distances, floor


def _jaccard_distance(a: frozenset[str], b: frozenset[str]) -> float | None:
    union = a | b
    if not union:
        # Neither recalled anything. That is not a distance of zero and not
        # one; it is a pair the measure could not read, and it is left out
        # rather than given a number.
        return None
    return 1.0 - len(a & b) / len(union)


def behavioural_floor(
    samples: dict[str, ClassSamples],
    classes: Sequence[PerceptClass],
) -> dict[str, float]:
    """How differently the same class recalls when presented twice.

    The floor the between-class distances have to clear. A floor of zero with
    between-class distances also at zero says retrieval does not depend on the
    percept at all, which is a fact about the channel rather than about the
    geometry.
    """
    out: dict[str, float] = {}
    for cls in classes:
        rows = samples.get(cls.name)
        if rows is None:
            continue
        values = [
            value
            for a, b in zip(rows.retrieved, rows.retrieved_b, strict=True)
            if (value := _jaccard_distance(a, b)) is not None
        ]
        if values:
            out[cls.name] = float(np.mean(values))
    return out


def behavioural_geometry(
    samples: dict[str, ClassSamples],
    classes: Sequence[PerceptClass],
) -> tuple[dict[tuple[int, int], float], dict[str, float]]:
    """One minus the overlap of what each class brought back, averaged per anchor.

    Paired by anchor rather than pooled, because pooling would let the spread
    between anchors stand in for the difference between classes, and the
    internal geometry conditions on the anchor.
    """
    names = [cls.name for cls in classes]
    distances: dict[tuple[int, int], float] = {}
    coverage: dict[str, float] = {}
    for name in names:
        rows = samples.get(name)
        sets = [] if rows is None else rows.retrieved
        coverage[name] = (
            0.0 if not sets else float(np.mean([1.0 if s else 0.0 for s in sets]))
        )
    for i, left_name in enumerate(names):
        left_sets = samples[left_name].retrieved if left_name in samples else []
        for j in range(i + 1, len(names)):
            right_sets = samples[names[j]].retrieved if names[j] in samples else []
            paired = min(len(left_sets), len(right_sets))
            if paired == 0:
                continue
            values = [
                value
                for index in range(paired)
                if (value := _jaccard_distance(left_sets[index], right_sets[index])) is not None
            ]
            if values:
                distances[(i, j)] = float(np.mean(values))
    return distances, coverage
