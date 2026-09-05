"""core/science/retrieval_latency.py — a latency law fitted to Aura, not borrowed.

``core/cognition/actr_activation.py`` did the honest thing and it is worth
saying plainly: it implemented ACT-R's retrieval-latency law, measured whether
it predicts Aura's recall timing, found r-squared of about 0.000037, and wrote
the refutation into the source instead of shipping the equation. ACT-R's law
describes a race between declarative chunks. Aura's recall is a ranked scan
over an embedding store, and there is no reason those should share a functional
form.

So the transferable half of the ACT-R timing programme is not the equation. It
is the *discipline*: latency is part of the theory, and a memory system that
cannot predict its own retrieval time does not understand its own retrieval.

The law here is fitted to the work Aura actually does::

    seconds = intercept + a·candidates + b·store_hops + c·embedding_calls

Three terms, all of them things the retrieval path really spends time on, all
of them countable at call time. It is linear because retrieval work is
approximately additive and because a curve fitted to a few hundred recalls
would describe the noise. If a term does not earn its coefficient, the fit
reports it and the term goes.

What makes it a law rather than a regression
--------------------------------------------
:meth:`LatencyLaw.intervene` is the test card 002 asks for. Predicting held-out
latency is a correlation. Changing the candidate count and having the measured
time move in the predicted direction, with backend and store held fixed, is
the causal claim — and it is the one that failed for the imported equation.

Error typing
------------
:func:`classify_error` is card 006's half that does not need human data. When
recall returns the wrong thing, the interesting question is which wrong thing:
a COMMISSION (something similar was returned) and an OMISSION (nothing was
returned) come from different failures and calling them both "a miss" hides
the distinction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "RetrievalObservation",
    "LatencyLaw",
    "ErrorKind",
    "classify_error",
    "fit",
]

#: Recalls needed before a fit is reported at all. Below this the coefficients
#: are the sample.
MIN_OBSERVATIONS = 40


@dataclass(frozen=True, slots=True)
class RetrievalObservation:
    """One recall, with the work it did and the time it took."""

    seconds: float
    candidates: int
    store_hops: int = 1
    embedding_calls: int = 0
    backend: str = ""
    hit: bool = True


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting. None when singular."""
    n = len(matrix)
    augmented = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for row in range(column + 1, n):
            factor = augmented[row][column] / augmented[column][column]
            for k in range(column, n + 1):
                augmented[row][k] -= factor * augmented[column][k]
    out = [0.0] * n
    for row in range(n - 1, -1, -1):
        total = augmented[row][n] - sum(augmented[row][k] * out[k] for k in range(row + 1, n))
        out[row] = total / augmented[row][row]
    return out


@dataclass(frozen=True, slots=True)
class LatencyLaw:
    """Coefficients, and how much of the variance they actually explain."""

    intercept: float
    per_candidate: float
    per_store_hop: float
    per_embedding_call: float
    r_squared: float
    n: int
    backend: str = ""

    def predict(self, observation: RetrievalObservation) -> float:
        return (
            self.intercept
            + self.per_candidate * observation.candidates
            + self.per_store_hop * observation.store_hops
            + self.per_embedding_call * observation.embedding_calls
        )

    @property
    def explains_anything(self) -> bool:
        """Whether this law beats predicting the mean.

        The threshold is deliberately low. The imported ACT-R law scored
        0.000037 here; anything that clears 0.1 is describing something real
        even if it is not describing it well.
        """
        return self.r_squared >= 0.1

    def intervene(
        self, base: RetrievalObservation, *, candidates: int
    ) -> dict[str, Any]:
        """What this law says happens if the candidate count changes.

        The direction is the falsifiable part. A law whose predicted direction
        is wrong is refuted by one well-controlled intervention, which is more
        than a held-out r-squared can say.
        """
        changed = RetrievalObservation(
            seconds=0.0,
            candidates=candidates,
            store_hops=base.store_hops,
            embedding_calls=base.embedding_calls,
            backend=base.backend,
        )
        before, after = self.predict(base), self.predict(changed)
        return {
            "from_candidates": base.candidates,
            "to_candidates": candidates,
            "predicted_before": before,
            "predicted_after": after,
            "predicted_delta": after - before,
            "direction": "up" if after > before else "down" if after < before else "flat",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "intercept": self.intercept,
            "per_candidate": self.per_candidate,
            "per_store_hop": self.per_store_hop,
            "per_embedding_call": self.per_embedding_call,
            "r_squared": self.r_squared,
            "n": self.n,
            "backend": self.backend,
            "explains_anything": self.explains_anything,
        }


def fit(observations: Sequence[RetrievalObservation], *, backend: str = "") -> LatencyLaw | None:
    """Least squares over the three work terms. None when there is too little data.

    Returning ``None`` rather than a law with wide coefficients is the point:
    a law fitted to thirty recalls is a description of thirty recalls, and the
    caller has to know the difference.
    """
    rows = [o for o in observations if not backend or o.backend == backend]
    if len(rows) < MIN_OBSERVATIONS:
        return None
    columns = [
        [1.0] * len(rows),
        [float(o.candidates) for o in rows],
        [float(o.store_hops) for o in rows],
        [float(o.embedding_calls) for o in rows],
    ]
    # A term that never varies cannot be told apart from the intercept. Dropping
    # it is the honest handling: the coefficient is not small, it is unknowable
    # from this data, and it is reported as zero rather than fitted to noise.
    used = [0] + [i for i in (1, 2, 3) if len(set(columns[i])) > 1]
    design = [[columns[i][r] for i in used] for r in range(len(rows))]
    target = [o.seconds for o in rows]
    size = len(used)
    normal = [[sum(design[r][i] * design[r][j] for r in range(len(rows))) for j in range(size)]
              for i in range(size)]
    moment = [sum(design[r][i] * target[r] for r in range(len(rows))) for i in range(size)]
    fitted = _solve(normal, moment)
    if fitted is None:
        return None
    solution = [0.0, 0.0, 0.0, 0.0]
    for slot, value in zip(used, fitted, strict=True):
        solution[slot] = value

    mean = sum(target) / len(target)
    total = sum((y - mean) ** 2 for y in target)
    residual = sum(
        (target[r] - sum(solution[i] * design[r][i] for i in range(size))) ** 2
        for r in range(len(rows))
    )
    r_squared = 0.0 if total <= 0 else max(0.0, 1.0 - residual / total)
    return LatencyLaw(
        intercept=solution[0],
        per_candidate=solution[1],
        per_store_hop=solution[2],
        per_embedding_call=solution[3],
        r_squared=r_squared,
        n=len(rows),
        backend=backend,
    )


class ErrorKind(StrEnum):
    """Which way a recall went wrong. Both are misses and they are not alike."""

    #: The right thing came back.
    CORRECT = "correct"
    #: Something came back and it was wrong. Interference, not absence.
    COMMISSION = "commission"
    #: Nothing came back. The item was not reachable at all.
    OMISSION = "omission"
    #: Something came back, was wrong, and nothing similar was in the store —
    #: the store answered a question it should have declined.
    FABRICATION = "fabrication"


def classify_error(
    *, returned: str | None, expected: str, nearest_similarity: float = 0.0
) -> ErrorKind:
    """Type a recall failure before the answer is graded.

    ``nearest_similarity`` is how close the store's best candidate was to the
    expected item. A wrong answer with a near neighbour is interference; a
    wrong answer with nothing near it is the store reaching past its own
    evidence, which is a different defect with a different fix.
    """
    if returned is None or returned == "":
        return ErrorKind.OMISSION
    if returned == expected:
        return ErrorKind.CORRECT
    return ErrorKind.COMMISSION if nearest_similarity >= 0.5 else ErrorKind.FABRICATION
