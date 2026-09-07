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

Three things stop this from being a number that always comes out positive.

The error is summed rather than averaged, so a one-column half cannot count as
much as a forty-column half.

Each domain is reduced to a few principal components before anything is
fitted. Without that, the intact model has a hundred and eleven inputs and each
cut model has a fraction of them, and on a few thousand rows the wide model
overfits and loses to the narrow one — the first version of this file returned
a confidently negative score for exactly that reason, which reads as "cutting
helps" and means "the comparison was unfair". The components are computed on
the training rows only.

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


def _reduce(block: np.ndarray, train: slice, k: int) -> np.ndarray:
    """Top-k principal directions of one domain, fitted on training rows only."""
    reference = block[train]
    spread = reference.std(axis=0)
    keep = spread > 1e-9
    if not keep.any():
        return np.zeros((block.shape[0], 0))
    centre = reference[:, keep].mean(axis=0)
    scaled = (block[:, keep] - centre) / spread[keep]
    if scaled.shape[1] <= k:
        return scaled
    reference_scaled = (reference[:, keep] - centre) / spread[keep]
    _, _, vectors = np.linalg.svd(reference_scaled, full_matrices=False)
    return scaled @ vectors[:k].T


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
        source[key] = _reduce(now[:, columns], train, COMPONENTS)
        target_of[key] = _reduce(nxt[:, columns], train, COMPONENTS)
    all_source = np.hstack([source[key] for key in live])

    self_fit: dict[tuple[str, ...], tuple[float, float]] = {}
    full_fit: dict[tuple[str, ...], float] = {}

    def measure(block: tuple[str, ...]) -> tuple[float, float]:
        if block in self_fit:
            return self_fit[block]
        inputs = np.hstack([source[key] for key in block])
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
