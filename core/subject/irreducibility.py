"""How much is lost when the core is forced to run as two systems.

Integration in the weak sense — this changes that — is cheap and almost
everything has it. The question worth asking is whether the transition law
factorises:

    P(K_{t+1} | K_t) ~= P(A_{t+1} | A_t) P(B_{t+1} | B_t)

for some way of cutting the core in two. If it does for even one cut, the
system is two systems that share a process table.

So every bipartition of the domains gets fitted twice: once with the whole
previous state available, once with each half allowed to see only itself. The
same targets, the same estimator, the same held-out rows, so the difference is
the cross-partition information and nothing else. The score is the fraction of
the cut model's error that the intact model recovers,

    Phi_do(A|B) = (L_cut - L_full) / L_cut

and the system's score is the minimum over cuts, because a chain is not strong
at its strongest link.

Five things stop this from being a number that always comes out positive.

The error is summed rather than averaged, so a one-column half cannot count as
much as a forty-column half.

Each domain is reduced to a few principal components before anything is
fitted. Without that, the intact model has a hundred and eleven inputs and each
cut model has a fraction of them, and on a few thousand rows the wide model
overfits and loses to the narrow one — the first version of this file returned
a confidently negative score for exactly that reason, which reads as "cutting
helps" and means "the comparison was unfair". The components are computed on
the training rows only.

What is predicted is the change, not the level. Most of these columns barely
move from one frame to the next, so predicting K_{t+1} from K_t is a task whose
answer is "the same as last time": the intact model scored a held-out loss of
0.025, and the comparison between it and a cut model was a comparison of noise
inside the remaining two and a half percent. Targeting K_{t+1} - K_t removes
the part of the answer that is the question restated and leaves exactly what
the transition law did.

Both sides also see which phase produced the transition. A frame-to-frame step
inside a turn is not one transition law but thirty-one of them — the step from
frame nine to ten is always the same phase doing the same thing — and one linear
model fitted across all of them is fitting the pipeline's running order as
though it were noise. Held out, that misspecification hurts the intact model
more than the cut ones, because the intact model has more inputs to waste on
it, and the score comes out negative on a system that is genuinely coupled. The
phase identity is a covariate on every fit, so what is left to explain is what
the domains did.

And the whole computation runs unchanged over shuffled and replayed
recordings, where the cross-partition information is gone by construction and
the score has to collapse.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.subject.estimate import fit_predict, split_rows
from core.subject.recording import Recording
from core.subject.state import DOMAINS

__all__ = ["COMPONENTS", "PartitionReport", "phi_do", "transition_rows"]

#: Principal components kept per domain. Four is enough to carry a domain's
#: shape and small enough that ten of them together stay well inside the
#: number of rows a run produces.
COMPONENTS: int = 4


def phase_covariates(recording: Recording, rows: np.ndarray) -> np.ndarray:
    """One-hot of which phase produced each transition.

    Given to both sides of every comparison, so it cannot favour either. What
    it removes is the pipeline's running order, which is real structure and is
    not what irreducibility is asking about.
    """
    tags = list(recording.tags)
    if not any(tags):
        return np.zeros((rows.size, 0))
    names = sorted({tags[index] for index in rows})
    index_of = {name: position for position, name in enumerate(names)}
    out = np.zeros((rows.size, len(names)), dtype=np.float64)
    for row, index in enumerate(rows):
        out[row, index_of[tags[index]]] = 1.0
    return out


def transition_rows(recording: Recording, condition: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Consecutive frames as (K_t, K_{t+1}).

    Frames are consecutive in the life, so a turn boundary is a transition
    like any other. Restricting to one condition keeps only pairs where both
    frames were recorded under it, which drops the boundary pairs rather than
    pretending the condition did not change between them.
    """
    if condition is None:
        keep = np.arange(recording.frames - 1)
    else:
        flags = np.array([name == condition for name in recording.conditions])
        keep = np.array(
            [index for index in range(recording.frames - 1) if flags[index] and flags[index + 1]],
            dtype=np.int64,
        )
    if keep.size == 0:
        return np.zeros((0, recording.width)), np.zeros((0, recording.width))
    return recording.x[keep], recording.x[keep + 1]


def _transition_index(recording: Recording, condition: str | None = None) -> np.ndarray:
    if condition is None:
        return np.arange(recording.frames - 1)
    flags = np.array([name == condition for name in recording.conditions])
    return np.array(
        [index for index in range(recording.frames - 1) if flags[index] and flags[index + 1]],
        dtype=np.int64,
    )


@dataclass
class PartitionReport:
    """Every cut that was tried, and the cheapest one."""

    phi: float
    best_cut: tuple[tuple[str, ...], tuple[str, ...]]
    loss_full: float
    loss_cut: float
    scores: dict[str, float] = field(default_factory=dict)
    domains: tuple[str, ...] = ()
    pairs: int = 0
    degenerate: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        ranked = sorted(self.scores.items(), key=lambda kv: kv[1])
        return {
            "phi_do": round(self.phi, 6),
            "best_cut": ["".join(self.best_cut[0]), "".join(self.best_cut[1])],
            "loss_full": round(self.loss_full, 6),
            "loss_cut": round(self.loss_cut, 6),
            "domains": list(self.domains),
            "pairs": self.pairs,
            "cheapest_cuts": [{"cut": k, "phi": round(v, 6)} for k, v in ranked[:8]],
            "dearest_cuts": [{"cut": k, "phi": round(v, 6)} for k, v in ranked[-4:]],
            "degenerate": self.degenerate,
            "note": self.note,
        }


def _block_columns(recording: Recording, block: tuple[str, ...]) -> np.ndarray:
    return np.concatenate(
        [np.arange(recording.slices[key].start, recording.slices[key].stop) for key in block]
    )


def _basis(block: np.ndarray, train: slice, k: int) -> Any:
    """A projection into one domain's top-k directions, fitted on training rows.

    Returned as a function rather than an array so the same coordinates can be
    applied to the frame before and the frame after; projecting each end
    separately would make the difference between two unrelated bases.
    """
    reference = block[train]
    spread = reference.std(axis=0)
    keep = spread > 1e-9
    if not keep.any():
        return lambda values: np.zeros((values.shape[0], 0))
    centre = reference[:, keep].mean(axis=0)
    scale = spread[keep]
    if int(keep.sum()) <= k:
        return lambda values: (values[:, keep] - centre) / scale
    reference_scaled = (reference[:, keep] - centre) / scale
    _, _, vectors = np.linalg.svd(reference_scaled, full_matrices=False)
    directions = vectors[:k].T
    return lambda values: ((values[:, keep] - centre) / scale) @ directions


def phi_do(
    recording: Recording,
    *,
    condition: str | None = None,
    domains: tuple[str, ...] | None = None,
) -> PartitionReport:
    """The minimum over bipartitions of the loss the cut costs."""
    live = domains if domains is not None else recording.live_domains()
    live = tuple(key for key in DOMAINS if key in live)
    now, nxt = transition_rows(recording, condition)
    where = _transition_index(recording, condition)
    if len(live) < 2 or now.shape[0] < 40:
        return PartitionReport(
            phi=0.0,
            best_cut=(live[:1], live[1:]),
            loss_full=1.0,
            loss_cut=1.0,
            domains=live,
            pairs=int(now.shape[0]),
            degenerate=True,
            note="not enough moving domains or transitions to cut anything",
        )

    train, validate, test = split_rows(now.shape[0])

    # Reduce first, then fit. Both sides of every comparison see the same
    # representation of the same domains, so what differs between them is which
    # domains they may look at and nothing else.
    source: dict[str, np.ndarray] = {}
    target_of: dict[str, np.ndarray] = {}
    for key in live:
        columns = _block_columns(recording, (key,))
        basis = _basis(now[:, columns], train, COMPONENTS)
        source[key] = basis(now[:, columns])
        # The same basis applied to both ends, so the target is the movement of
        # the domain in its own coordinates rather than the difference between
        # two unrelated projections.
        target_of[key] = basis(nxt[:, columns]) - basis(now[:, columns])
    covariates = phase_covariates(recording, where + 1)
    all_source = np.hstack([source[key] for key in live] + [covariates])

    self_fit: dict[tuple[str, ...], tuple[float, float]] = {}
    full_fit: dict[tuple[str, ...], float] = {}

    def measure(block: tuple[str, ...]) -> tuple[float, float]:
        if block in self_fit:
            return self_fit[block]
        inputs = np.hstack([source[key] for key in block] + [covariates])
        target = np.hstack([target_of[key] for key in block])
        own = fit_predict(inputs, target, train=train, validate=validate, test=test)
        whole = fit_predict(all_source, target, train=train, validate=validate, test=test)
        self_fit[block] = (own.sse, own.base)
        full_fit[block] = whole.sse
        return self_fit[block]

    scores: dict[str, float] = {}
    best: tuple[float, tuple[tuple[str, ...], tuple[str, ...]], float, float] | None = None
    index = list(range(len(live)))
    for size in range(1, len(live) // 2 + 1):
        for chosen in itertools.combinations(index, size):
            side_a = tuple(live[i] for i in chosen)
            side_b = tuple(live[i] for i in index if i not in set(chosen))
            if size == len(live) - size and side_a > side_b:
                continue  # each cut once, not twice with the halves swapped
            sse_a, base_a = measure(side_a)
            sse_b, base_b = measure(side_b)
            base = base_a + base_b
            if base <= 0.0:
                continue
            loss_cut = (sse_a + sse_b) / base
            loss_full = (full_fit[side_a] + full_fit[side_b]) / base
            phi = 0.0 if loss_cut <= 0.0 else (loss_cut - loss_full) / loss_cut
            key = f"{''.join(side_a)}|{''.join(side_b)}"
            scores[key] = phi
            if best is None or phi < best[0]:
                best = (phi, (side_a, side_b), loss_full, loss_cut)

    if best is None:
        return PartitionReport(
            phi=0.0,
            best_cut=(live[:1], live[1:]),
            loss_full=1.0,
            loss_cut=1.0,
            domains=live,
            pairs=int(now.shape[0]),
            degenerate=True,
            note="no cut had a target with any variance",
        )
    return PartitionReport(
        phi=best[0],
        best_cut=best[1],
        loss_full=best[2],
        loss_cut=best[3],
        scores=scores,
        domains=live,
        pairs=int(now.shape[0]),
    )
