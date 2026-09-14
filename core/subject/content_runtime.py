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

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


from core.state.percepts import DEFAULT_INTENSITY, PERCEPT_EMOTIONS, emit_percept
from core.subject.content import PerceptClass
from core.subject.intrinsic_v25 import crossfit_fisher_rao
from core.subject.state import perturb, perturb_organs

__all__ = [
    "EMOTIONS",
    "KINDS",
    "ClassSamples",
    "behavioural_geometry",
    "internal_geometry",
    "grid",
    "present",
    "sample_classes",
]

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
                intensity=DEFAULT_INTENSITY,
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
    """What one percept class produced, per anchor."""

    name: str
    futures: list[np.ndarray]
    retrieved: list[frozenset[str]]
    context: list[np.ndarray]


def _retrieved(runtime: Any) -> frozenset[str]:
    cognition = getattr(getattr(runtime, "state", None), "cognition", None)
    items = list(getattr(cognition, "long_term_memory", []) or [])
    return frozenset(str(item) for item in items)


async def sample_classes(
    runtime: Any,
    anchors: Sequence[Any],
    conditions: Sequence[Any],
    classes: Sequence[PerceptClass],
    *,
    turns: int = 1,
    lag: int = 1,
    displace: tuple[str, float] | None = None,
) -> dict[str, ClassSamples]:
    """Present every class from every anchor, and read both geometries' inputs.

    `displace` moves one domain before the percept arrives. That is how the
    manifold is perturbed: the same classes are presented into a state that has
    been pushed, and both geometries are recomputed on what comes back.
    """
    from dataclasses import replace

    out: dict[str, ClassSamples] = {
        cls.name: ClassSamples(cls.name, [], [], []) for cls in classes
    }
    for anchor in anchors:
        for condition in conditions:
            for cls in classes:
                runtime.restore(anchor.snapshot)
                if displace is not None:
                    domain, delta = displace
                    perturb(runtime.state, domain, delta, ontogeny=runtime.ontogeny)
                    await perturb_organs(
                        runtime.organs, domain, delta, state=runtime.state
                    )
                shown = replace(condition, prepare=present(cls))
                rows: list[Any] = []
                for _ in range(turns):
                    rows.extend(await runtime.turn_once(shown))
                if not rows:
                    continue
                index = min(max(1, lag) - 1, len(rows) - 1)
                out[cls.name].futures.append(
                    np.asarray(rows[index].vector(), dtype=np.float64)
                )
                out[cls.name].retrieved.append(_retrieved(runtime))
                out[cls.name].context.append(np.asarray(anchor.current, dtype=np.float64))
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

    blocks = {name: block(name) for name in names}
    for i, left_name in enumerate(names):
        left = blocks[left_name]
        if left is None:
            continue
        # The floor for this class: its own samples against its own samples,
        # split by the same estimator. Anything the estimator reads off finite
        # data rather than off the classes shows up here.
        half = len(left) // 2
        if half >= folds:
            same = crossfit_fisher_rao(
                left[:half], left[half : 2 * half], folds=folds, seed=seed
            )
            floor[(i, i)] = float(same.distance_sq)
        for j in range(i + 1, len(names)):
            right = blocks[names[j]]
            if right is None:
                continue
            estimate = crossfit_fisher_rao(left, right, folds=folds, seed=seed)
            distances[(i, j)] = float(estimate.distance_sq)
    return distances, floor


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
            values: list[float] = []
            for index in range(paired):
                a, b = left_sets[index], right_sets[index]
                union = a | b
                if not union:
                    # Neither recalled anything. That is not a distance of zero
                    # and not one; it is a pair the measure could not read, and
                    # it is left out rather than given a number.
                    continue
                values.append(1.0 - len(a & b) / len(union))
            if values:
                distances[(i, j)] = float(np.mean(values))
    return distances, coverage
