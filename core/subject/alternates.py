"""The battery's two lower-bound criteria, measured again with other estimators.

A result that holds only for the estimator it was designed with is a result
about the estimator. P54.25 asks that the verdict survive reasonable
alternatives, so the two criteria the battery reads as a lower bound are
measured here a second way, on the same recording and the same rows.

Irreducibility is re-scored with the settings its estimator had to choose: how
many principal components stand for a domain, and how many forward-chaining
folds the bound is read over. Nothing else about the estimator changes, so a
verdict that moves with these moved because of them.

Synergy is re-scored with a different mutual information estimator. The
battery's is the closed-form Gaussian one on a rank copula, which cannot see
dependence that is not linear in the ranks. The Kraskov, Stoegbauer and
Grassberger nearest-neighbour estimator (Physical Review E 69, 066138, 2004;
their algorithm 1) makes no distributional assumption and fails differently,
so agreement between the two means more than either. It is read against the
same kind of shifted null: both sources slid together against the target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

from core.subject.recording import Recording

__all__ = [
    "AlternateSynergy",
    "irreducibility_under",
    "ksg_mutual_information",
    "ksg_synergy",
]

#: The transitions the nearest-neighbour synergy was shown to tell a product
#: from a sum at: 2,400 turns, so 2,399 transitions, which is a 300-round run.
#: On the ISC-v2 known-answer shapes its raw synergy cleared its shifted null on
#: 3 of 3 seeds for a product and 0 of 3 for an additive target there, and on 1
#: of 3 for the product at 479. Below this it is reported as underpowered rather
#: than as a verdict.
KNOWN_ANSWER_ROWS: int = 2399

#: Neighbours in the Kraskov estimator. Their paper shows the bias and variance
#: trade over k and recommends small values; four is the middle of the range
#: they tested and the value most later work reports.
KSG_NEIGHBOURS: int = 4


def ksg_mutual_information(x: np.ndarray, y: np.ndarray, *, k: int = KSG_NEIGHBOURS, seed: int = 0) -> float:
    """I(X;Y) in nats by the first Kraskov estimator, with the max norm.

    A vanishing jitter, a ten-billionth of each column's spread, breaks the ties
    a discrete column produces, which the estimator's counts are not defined for.
    """
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    y = np.asarray(y, dtype=np.float64).reshape(len(y), -1)
    n = len(x)
    if n <= k + 1 or len(y) != n:
        return 0.0
    rng = np.random.default_rng(seed)

    def jittered(block: np.ndarray) -> np.ndarray:
        spread = block.std(axis=0)
        spread[spread <= 0.0] = 1.0
        return block + rng.normal(size=block.shape) * spread * 1e-10

    x, y = jittered(x), jittered(y)
    joint = np.hstack([x, y])
    distances, _ = cKDTree(joint).query(joint, k=k + 1, p=np.inf)
    radius = np.nextafter(distances[:, k], 0.0)
    in_x = cKDTree(x).query_ball_point(x, r=radius, p=np.inf, return_length=True) - 1
    in_y = cKDTree(y).query_ball_point(y, r=radius, p=np.inf, return_length=True) - 1
    value = digamma(k) + digamma(n) - float(np.mean(digamma(in_x + 1) + digamma(in_y + 1)))
    return float(value)


@dataclass
class AlternateSynergy:
    sources: tuple[str, str]
    target: str
    of: str
    synergy: float
    fraction: float
    null_q99: float
    raw_null_q99: float
    null_draws: int
    rows: int

    @property
    def clears_its_null(self) -> bool:
        """The raw synergy above the 99th percentile of its own shifted null.

        The battery's fraction bar of 0.10 is a bar on the Gaussian estimate.
        The nearest-neighbour estimate of the joint information carries a
        different bias, so its fractions sit lower for the same interaction:
        0.08 to 0.11 for the known-answer product. What transfers between
        estimators is the comparison against the estimator's own null, and the
        raw one is what the battery calls the stricter of its two.
        """
        return self.synergy > self.raw_null_q99

    @property
    def underpowered(self) -> bool:
        return self.rows < KNOWN_ANSWER_ROWS

    def as_dict(self) -> dict[str, Any]:
        return {
            "sources": list(self.sources),
            "target": self.target,
            "of": self.of,
            "synergy": round(self.synergy, 5),
            "synergy_fraction": round(self.fraction, 5),
            "null_q99_fraction": round(self.null_q99, 5),
            "raw_null_q99": round(self.raw_null_q99, 5),
            "null_draws": self.null_draws,
            "rows": self.rows,
            "clears_its_null": self.clears_its_null,
            "underpowered": self.underpowered,
        }


def _decomposition(a: np.ndarray, b: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    joint = ksg_mutual_information(np.hstack([a, b]), y, seed=seed)
    mi_a = ksg_mutual_information(a, y, seed=seed + 1)
    mi_b = ksg_mutual_information(b, y, seed=seed + 2)
    value = joint - mi_a - mi_b + min(mi_a, mi_b)
    return value, joint


def ksg_synergy(
    recording: Recording,
    source_a: str,
    source_b: str,
    target: str,
    *,
    of: str = "change",
    draws: int = 200,
    seed: int = 0,
) -> AlternateSynergy:
    """The battery's minimum-information synergy, with the Kraskov estimator in place of the Gaussian one."""
    from core.subject.synergy import _components, _copula_normal

    raw_a = _components(recording.domain(source_a)[:-1])
    raw_b = _components(recording.domain(source_b)[:-1])
    following = recording.domain(target)
    raw_y = _components(following[1:] - following[:-1] if of == "change" else following[1:])
    a, b, y = (_copula_normal(block) for block in (raw_a, raw_b, raw_y))
    rows = a.shape[0]
    value, joint = _decomposition(a, b, y, seed)
    fraction = value / joint if joint > 1e-9 else 0.0
    rng = np.random.default_rng(seed)
    nulls: list[float] = []
    raw_nulls: list[float] = []
    for draw in range(draws):
        shift = int(rng.integers(rows // 8, rows - rows // 8)) if rows > 16 else 1
        null_value, null_joint = _decomposition(np.roll(a, shift, axis=0), np.roll(b, shift, axis=0), y, seed + 10 + draw)
        raw_nulls.append(null_value)
        # The battery's own guard: a fraction is only taken while the null
        # still has a tenth of the real joint information to divide by.
        nulls.append(null_value / null_joint if null_joint > max(1e-6, 0.1 * joint) else 0.0)
    return AlternateSynergy(
        sources=(source_a, source_b),
        target=target,
        of=of,
        synergy=value,
        fraction=float(fraction),
        null_q99=float(np.quantile(nulls, 0.99)) if nulls else 1.0,
        raw_null_q99=float(np.quantile(raw_nulls, 0.99)) if raw_nulls else float("inf"),
        null_draws=len(nulls),
        rows=rows,
    )


@dataclass
class IrreducibilityUnder:
    components: int
    folds: int
    phi: float
    lower_bound: float
    standard_error: float
    best_cut: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "components": self.components,
            "folds": self.folds,
            "phi_do": round(self.phi, 6),
            "lower_bound": round(self.lower_bound, 6),
            "standard_error": round(self.standard_error, 6),
            "best_cut": self.best_cut,
        }


def irreducibility_under(recording: Recording, *, components: int, folds: int) -> IrreducibilityUnder:
    """Irreducibility with another component count and fold count, everything else as the battery has it."""
    from core.subject.irreducibility import LOWER_BOUND_Z, phi_do

    report = phi_do(recording, components=components, folds=folds)
    return IrreducibilityUnder(
        components=components,
        folds=folds,
        phi=float(report.phi),
        lower_bound=float(report.phi - LOWER_BOUND_Z * report.standard_error),
        standard_error=float(report.standard_error),
        best_cut=["".join(report.best_cut[0]), "".join(report.best_cut[1])],
    )
