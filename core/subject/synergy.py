"""Information that exists only in the combination.

Two sources can carry the same thing about a target, or different things, or
something neither carries alone. The last is synergy, and it is the one that
distinguishes a state whose parts are read together from a state whose parts
are read separately and added up.

The decomposition used is the minimum-information one: redundancy is the
smaller of the two single-source informations, which for jointly Gaussian
variables is the standard choice and has the advantage of being computable in
closed form from a covariance matrix rather than estimated from a histogram
nobody has enough samples to fill.

    Syn = I(X1,X2;Y) - I(X1;Y) - I(X2;Y) + min(I(X1;Y), I(X2;Y))

Each domain is reduced to a few principal components first. Forty columns
against a few thousand rows makes a covariance matrix that is nearly singular,
and a determinant near zero can report any amount of information at all.

Because a positive number here would be easy to produce by accident, the same
computation runs against a null built by sliding one source in time against
the target. The shift keeps every marginal, every autocorrelation and every
within-source relationship, and destroys only the alignment that synergy is
supposed to be about.

The linear estimator cannot see an interaction, so a second measurement asks
the question a different way: does adding the products of the two sources'
components improve held-out prediction over the two sources side by side? That
one is model-based and coarse, and it is here because it fails differently
from the Gaussian estimate, so agreement between them means more than either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.subject.estimate import fit_predict, split_rows
from core.subject.recording import Recording

__all__ = ["SynergyReport", "synergy", "synergy_suite"]

#: Components kept per domain. Three is enough to carry the shape of a domain
#: and small enough that the joint covariance stays invertible.
COMPONENTS: int = 3

#: Shifts used for the null. Each one is a whole-trajectory circular slide.
NULL_DRAWS: int = 200


def _components(block: np.ndarray, k: int = COMPONENTS) -> np.ndarray:
    spread = block.std(axis=0)
    keep = spread > 1e-9
    if not keep.any():
        return np.zeros((block.shape[0], 1))
    centred = (block[:, keep] - block[:, keep].mean(axis=0)) / spread[keep]
    if centred.shape[1] <= k:
        return centred
    _, _, vectors = np.linalg.svd(centred, full_matrices=False)
    return centred @ vectors[:k].T


def _gaussian_mi(x: np.ndarray, y: np.ndarray) -> float:
    """I(X;Y) for jointly Gaussian blocks, from log determinants."""
    if x.size == 0 or y.size == 0:
        return 0.0
    joint = np.hstack([x, y])
    ridge = 1e-6
    cxx = np.cov(x, rowvar=False).reshape(x.shape[1], x.shape[1]) + ridge * np.eye(x.shape[1])
    cyy = np.cov(y, rowvar=False).reshape(y.shape[1], y.shape[1]) + ridge * np.eye(y.shape[1])
    cjj = np.cov(joint, rowvar=False) + ridge * np.eye(joint.shape[1])
    sign_x, log_x = np.linalg.slogdet(cxx)
    sign_y, log_y = np.linalg.slogdet(cyy)
    sign_j, log_j = np.linalg.slogdet(cjj)
    if min(sign_x, sign_y, sign_j) <= 0:
        return 0.0
    return max(0.0, 0.5 * float(log_x + log_y - log_j))


@dataclass
class SynergyReport:
    sources: tuple[str, str]
    target: str
    joint: float
    unique_a: float
    unique_b: float
    redundancy: float
    synergy: float
    normalised: float
    null_q99: float
    interaction_gain: float
    rows: int

    @property
    def passes(self) -> bool:
        return self.normalised >= 0.10 and self.normalised > self.null_q99

    def as_dict(self) -> dict[str, Any]:
        return {
            "sources": list(self.sources),
            "target": self.target,
            "joint_information": round(self.joint, 4),
            "redundancy": round(self.redundancy, 4),
            "unique": [round(self.unique_a, 4), round(self.unique_b, 4)],
            "synergy": round(self.synergy, 4),
            "synergy_fraction": round(self.normalised, 4),
            "null_q99_fraction": round(self.null_q99, 4),
            "interaction_gain": round(self.interaction_gain, 4),
            "rows": self.rows,
            "passes": self.passes,
        }


def _interaction_gain(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
    """Held-out improvement from letting the two sources multiply."""
    rows = a.shape[0]
    if rows < 80:
        return 0.0
    train, validate, test = split_rows(rows)
    side_by_side = np.hstack([a, b])
    products = np.einsum("ti,tj->tij", a, b).reshape(rows, -1)
    plain = fit_predict(side_by_side, y, train=train, validate=validate, test=test)
    crossed = fit_predict(
        np.hstack([side_by_side, products]),
        y,
        train=train,
        validate=validate,
        test=test,
        own_width=side_by_side.shape[1],
    )
    if plain.loss <= 1e-12:
        return 0.0
    return float((plain.loss - crossed.loss) / plain.loss)


def synergy(
    recording: Recording,
    source_a: str,
    source_b: str,
    target: str,
    *,
    seed: int = 0,
) -> SynergyReport:
    """Syn(A_t, B_t ; Y_{t+1}) with a shifted null and an interaction check."""
    a = _components(recording.domain(source_a)[:-1])
    b = _components(recording.domain(source_b)[:-1])
    y = _components(recording.domain(target)[1:])
    rows = a.shape[0]
    if rows < 40 or min(a.shape[1], b.shape[1], y.shape[1]) < 1:
        return SynergyReport((source_a, source_b), target, 0, 0, 0, 0, 0, 0, 1, 0, rows)

    joint = _gaussian_mi(np.hstack([a, b]), y)
    mi_a = _gaussian_mi(a, y)
    mi_b = _gaussian_mi(b, y)
    redundancy = min(mi_a, mi_b)
    unique_a = mi_a - redundancy
    unique_b = mi_b - redundancy
    value = joint - redundancy - unique_a - unique_b
    fraction = value / joint if joint > 1e-9 else 0.0

    rng = np.random.default_rng(seed)
    nulls = np.empty(NULL_DRAWS, dtype=np.float64)
    for draw in range(NULL_DRAWS):
        shift = int(rng.integers(rows // 8, rows - rows // 8)) if rows > 16 else 1
        slid = np.roll(b, shift, axis=0)
        null_joint = _gaussian_mi(np.hstack([a, slid]), y)
        null_b = _gaussian_mi(slid, y)
        null_red = min(mi_a, null_b)
        null_value = null_joint - null_red - (mi_a - null_red) - (null_b - null_red)
        nulls[draw] = null_value / null_joint if null_joint > 1e-9 else 0.0

    return SynergyReport(
        sources=(source_a, source_b),
        target=target,
        joint=joint,
        unique_a=unique_a,
        unique_b=unique_b,
        redundancy=redundancy,
        synergy=value,
        normalised=float(fraction),
        null_q99=float(np.quantile(nulls, 0.99)),
        interaction_gain=_interaction_gain(a, b, y),
        rows=rows,
    )


#: The three triples named in the specification, plus one that closes the
#: self/action loop. Fixed before the run so a passing triple cannot be found
#: by searching all four hundred of them.
TRIPLES: tuple[tuple[str, str, str], ...] = (
    ("A", "S", "G"),
    ("P", "M", "W"),
    ("W", "A", "D"),
    ("S", "D", "C"),
)


def synergy_suite(recording: Recording, *, seed: int = 0) -> list[SynergyReport]:
    return [synergy(recording, a, b, y, seed=seed) for a, b, y in TRIPLES]
