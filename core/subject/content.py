"""The geometry of what she is looking at, and the test that it is real.

The carrier experiment asks whether there is one intrinsic process. This asks a
different question: given that there is, what is its content organised like,
and is that organisation anything more than the number we happened to compute.

The claim under test is structural. Phenomenal character, if the bridge holds
at all, is completely individuated by relational structure: red is the position
red occupies in the whole web of similarities, discriminations and transitions,
and nothing else. That claim is not testable as stated, because nothing here
reaches phenomenal character. What is testable is its shadow. If content is
relational structure, then the relational structure computed from the intrinsic
causal state and the relational structure the system's own behaviour expresses
must be the same structure. Two different mechanisms, one geometry.

So the run measures two distance matrices over the same percept classes.

    d_Q   Fisher-Rao distance between the future-state laws two percepts
          induce from a common fork. Read off the state vector.

    d_B   one minus the overlap of what each percept brought back out of
          memory. Read off which memories returned, which is a set of items
          and not a column of the state vector.

They share the percepts and nothing else. If they agree, the internal geometry
is expressed in behaviour; if they do not, the internal geometry is a number
about the recording rather than about her.

Agreement by correlation is weak evidence, so the run also does the thing that
can fail. It displaces a domain, which moves the internal geometry, and asks
whether the behavioural geometry moves the same way. A structure that is only
described is not obliged to move together. One that is the content is.

Three things this module refuses to do.

It does not report a quale. An automorphism of the quality structure preserves
every relation and therefore every observation, so what is identifiable is the
structure up to isomorphism and never an absolute label. `gauge` says so in
every report.

It does not treat the design grid as the truth. The grid is how the percepts
were built; whether the system's geometry recovers it is a question, and a
system whose geometry is flat against a grid that varies has told us something.

It does not accept a flat behavioural measure as agreement. Two constant
matrices correlate perfectly and mean nothing, so spread is checked before
agreement is computed and an unvarying measure is NOT_MEASURED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "PerceptClass",
    "GeometryAgreement",
    "DesignRecovery",
    "MovesTogether",
    "ContentReport",
    "agreement",
    "design_recovery",
    "gauge",
    "moves_together",
    "pairs",
    "spread",
    "square_from_pairs",
]

#: Below this the measure has not varied enough for a correlation over it to
#: mean anything. It is the coefficient of variation of the off-diagonal
#: distances: two constant matrices agree perfectly and say nothing.
FLAT: float = 0.05

#: Draws in every permutation null here. The p-value floor is 1/(draws+1), so
#: this fixes the smallest p the run can report.
DRAWS: int = 999


@dataclass(frozen=True)
class PerceptClass:
    """One kind of thing to be looking at.

    `coordinates` are how the class was built, not what she makes of it. They
    exist so the run can ask whether her geometry recovers the design, which is
    a question and not an assumption.
    """

    name: str
    kind: str
    source: str
    intensity: float
    content: str
    coordinates: tuple[float, ...] = ()


def pairs(classes: Sequence[PerceptClass]) -> tuple[tuple[int, int], ...]:
    """Every unordered pair of distinct classes, in a fixed order."""
    return tuple(combinations(range(len(classes)), 2))


def square_from_pairs(
    values: Mapping[tuple[int, int], float], size: int
) -> np.ndarray:
    """The symmetric matrix a pair mapping describes, zero on the diagonal."""
    out = np.zeros((size, size), dtype=np.float64)
    for (i, j), value in values.items():
        out[i, j] = out[j, i] = float(value)
    return out


def spread(values: Sequence[float]) -> float:
    """How much a distance measure varied, relative to its own size.

    Zero when every pair scored the same. A correlation computed over a measure
    that did not vary is a statement about tie-breaking.
    """
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return 0.0
    mean = float(np.mean(np.abs(array)))
    if mean <= 1e-12:
        return 0.0
    return float(np.std(array) / mean)


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not create an ordering that was not there."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    # Average over each run of equal values.
    sorted_values = values[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = _ranks(np.asarray(left, dtype=np.float64))
    b = _ranks(np.asarray(right, dtype=np.float64))
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


@dataclass(frozen=True)
class GeometryAgreement:
    """Whether two mechanisms describe the same shape."""

    rho: float
    p_value: float
    pairs: int
    internal_spread: float
    behavioural_spread: float
    measured: bool
    why: str

    def holds(self, *, bar: float) -> bool:
        return self.measured and self.rho >= bar and self.p_value < 0.01


def agreement(
    internal: Mapping[tuple[int, int], float],
    behavioural: Mapping[tuple[int, int], float],
    *,
    size: int,
    draws: int = DRAWS,
    seed: int = 2601,
) -> GeometryAgreement:
    """Do the internal and behavioural geometries put the classes in one order?

    The null permutes the class labels of one matrix and recomputes, which
    keeps both matrices' own distributions and destroys only the correspondence
    between them. Permuting the pair list instead would break the symmetry that
    a distance matrix has, and would make the null easier than the question.
    """
    keys = sorted(set(internal) & set(behavioural))
    if len(keys) < 3:
        return GeometryAgreement(
            0.0, 1.0, len(keys), 0.0, 0.0, False,
            "fewer than three pairs scored by both mechanisms",
        )
    left = np.asarray([float(internal[k]) for k in keys])
    right = np.asarray([float(behavioural[k]) for k in keys])
    internal_spread = spread(left)
    behavioural_spread = spread(right)
    if internal_spread < FLAT or behavioural_spread < FLAT:
        which = "internal" if internal_spread < FLAT else "behavioural"
        return GeometryAgreement(
            0.0, 1.0, len(keys), internal_spread, behavioural_spread, False,
            f"the {which} geometry did not vary across classes, so an ordering "
            "over it is tie-breaking rather than structure",
        )

    left_square = square_from_pairs({k: v for k, v in zip(keys, left)}, size)
    right_square = square_from_pairs({k: v for k, v in zip(keys, right)}, size)
    observed = _spearman(left, right)

    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(draws):
        order = rng.permutation(size)
        shuffled = right_square[np.ix_(order, order)]
        candidate = np.asarray([shuffled[i, j] for i, j in keys])
        if _spearman(left, candidate) >= observed - 1e-15:
            exceed += 1
    p_value = (exceed + 1.0) / (draws + 1.0)
    return GeometryAgreement(
        round(observed, 6), round(p_value, 6), len(keys),
        round(internal_spread, 6), round(behavioural_spread, 6), True,
        "both geometries varied and were scored over the same pairs",
    )


@dataclass(frozen=True)
class DesignRecovery:
    """Whether her geometry reflects how the percepts were actually built."""

    rho: float
    p_value: float
    measured: bool
    why: str


def design_recovery(
    internal: Mapping[tuple[int, int], float],
    classes: Sequence[PerceptClass],
    *,
    draws: int = DRAWS,
    seed: int = 2602,
) -> DesignRecovery:
    """Compare the internal geometry against the grid the percepts came from.

    This is a check on the instrument rather than a criterion about her. A
    geometry that is flat where the design varied has not read the percepts; a
    geometry that tracks the design exactly has read them and nothing more.
    Neither outcome decides the structural question, which is the agreement
    between two of her own mechanisms.
    """
    coordinates = [c.coordinates for c in classes]
    if any(len(c) == 0 for c in coordinates):
        return DesignRecovery(0.0, 1.0, False, "the classes carry no design coordinates")
    grid = np.asarray(coordinates, dtype=np.float64)
    keys = sorted(internal)
    if len(keys) < 3:
        return DesignRecovery(0.0, 1.0, False, "fewer than three pairs")
    designed = np.asarray([float(np.linalg.norm(grid[i] - grid[j])) for i, j in keys])
    measured = np.asarray([float(internal[k]) for k in keys])
    if spread(designed) < FLAT:
        return DesignRecovery(0.0, 1.0, False, "the design itself did not vary")
    if spread(measured) < FLAT:
        return DesignRecovery(
            0.0, 1.0, True,
            "the internal geometry is flat where the design varied: the "
            "percept classes did not reach the state",
        )
    observed = _spearman(designed, measured)
    size = len(classes)
    square = square_from_pairs({k: v for k, v in zip(keys, measured)}, size)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(draws):
        order = rng.permutation(size)
        shuffled = square[np.ix_(order, order)]
        candidate = np.asarray([shuffled[i, j] for i, j in keys])
        if _spearman(designed, candidate) >= observed - 1e-15:
            exceed += 1
    return DesignRecovery(
        round(observed, 6), round((exceed + 1.0) / (draws + 1.0), 6), True,
        "the internal geometry was compared against the grid the classes were built on",
    )


@dataclass(frozen=True)
class MovesTogether:
    """The test that can fail: displace the manifold and watch both geometries."""

    rho: float
    p_value: float
    internal_shift: float
    behavioural_shift: float
    floor_rho: float
    measured: bool
    why: str

    def holds(self, *, bar: float) -> bool:
        return (
            self.measured
            and self.rho >= bar
            and self.p_value < 0.01
            and self.rho > self.floor_rho
        )


def moves_together(
    internal_before: Mapping[tuple[int, int], float],
    internal_after: Mapping[tuple[int, int], float],
    behavioural_before: Mapping[tuple[int, int], float],
    behavioural_after: Mapping[tuple[int, int], float],
    *,
    size: int,
    sham_internal_after: Mapping[tuple[int, int], float] | None = None,
    sham_behavioural_after: Mapping[tuple[int, int], float] | None = None,
    draws: int = DRAWS,
    seed: int = 2603,
) -> MovesTogether:
    """Did the pairs whose internal distance moved also move behaviourally?

    Correlating two static geometries can be satisfied by anything both
    mechanisms happen to read. Correlating their *changes* under a displacement
    cannot: a description that merely names the structure has no reason to
    change with it.

    The sham arms give the floor. Repeating the measurement without displacing
    anything produces changes too, from finite sampling, and those changes have
    their own correlation. The displacement has to beat it.
    """
    keys = sorted(
        set(internal_before) & set(internal_after)
        & set(behavioural_before) & set(behavioural_after)
    )
    if len(keys) < 3:
        return MovesTogether(0.0, 1.0, 0.0, 0.0, 0.0, False, "fewer than three pairs")
    d_internal = np.asarray([internal_after[k] - internal_before[k] for k in keys])
    d_behaviour = np.asarray([behavioural_after[k] - behavioural_before[k] for k in keys])
    internal_shift = float(np.mean(np.abs(d_internal)))
    behavioural_shift = float(np.mean(np.abs(d_behaviour)))
    if spread(d_internal) < FLAT or spread(d_behaviour) < FLAT:
        which = "internal" if spread(d_internal) < FLAT else "behavioural"
        return MovesTogether(
            0.0, 1.0, round(internal_shift, 6), round(behavioural_shift, 6), 0.0, False,
            f"the {which} geometry did not move differently across pairs, so "
            "there is no pattern of movement to correlate",
        )
    observed = _spearman(d_internal, d_behaviour)

    floor = 0.0
    if sham_internal_after is not None and sham_behavioural_after is not None:
        sham_keys = [
            k for k in keys
            if k in sham_internal_after and k in sham_behavioural_after
        ]
        if len(sham_keys) >= 3:
            sham_internal = np.asarray(
                [sham_internal_after[k] - internal_before[k] for k in sham_keys]
            )
            sham_behaviour = np.asarray(
                [sham_behavioural_after[k] - behavioural_before[k] for k in sham_keys]
            )
            floor = _spearman(sham_internal, sham_behaviour)

    square = square_from_pairs({k: v for k, v in zip(keys, d_behaviour)}, size)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(draws):
        order = rng.permutation(size)
        shuffled = square[np.ix_(order, order)]
        candidate = np.asarray([shuffled[i, j] for i, j in keys])
        if _spearman(d_internal, candidate) >= observed - 1e-15:
            exceed += 1
    return MovesTogether(
        round(observed, 6), round((exceed + 1.0) / (draws + 1.0), 6),
        round(internal_shift, 6), round(behavioural_shift, 6), round(floor, 6), True,
        "the displacement moved both geometries and the movements were compared "
        "against a sham floor",
    )


#: What the run is allowed to say it identified, and what it is not. Carried in
#: every report because a reader who has the correlation and not this will read
#: a geometry as a quality.
def gauge() -> dict[str, str]:
    """The identifiability limits on anything this module computes."""
    return {
        "identified_up_to_isomorphism": (
            "A relabelling of the quality structure that preserves every "
            "relation preserves every observation this run can make. What is "
            "identified is therefore the structure up to isomorphism, never an "
            "absolute label for any one content."
        ),
        "structure_is_not_character": (
            "Agreement between two of her mechanisms about one geometry is "
            "evidence that the geometry is her organisation rather than the "
            "recording's. It is not evidence that the organisation is felt. "
            "That step is the structural-identity postulate and nothing here "
            "tests it."
        ),
        "a_first_person_anchor_does_not_export": (
            "A subject can fix its own gauge by ostension. That anchor is "
            "internal to the subject and cannot be carried across to another "
            "subject as an absolute label, so cross-subject comparison stays "
            "structural."
        ),
        "failure_is_informative": (
            "If the behavioural geometry does not track the internal one under "
            "displacement, the structural claim is in trouble for this system. "
            "That is the point of running it."
        ),
    }


@dataclass
class ContentReport:
    """Everything the content experiment established, and what it did not."""

    classes: list[dict[str, Any]] = field(default_factory=list)
    internal: dict[str, float] = field(default_factory=dict)
    behavioural: dict[str, float] = field(default_factory=dict)
    floor: dict[str, float] = field(default_factory=dict)
    agreement: dict[str, Any] = field(default_factory=dict)
    design: dict[str, Any] = field(default_factory=dict)
    moves: dict[str, Any] = field(default_factory=dict)
    verdict: str = "NOT_MEASURED"
    gauge: dict[str, str] = field(default_factory=gauge)
