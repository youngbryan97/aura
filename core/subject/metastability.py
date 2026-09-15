"""Whether the state settles into anything, and whether it ever leaves.

Four regimes are uninteresting and easy to mistake for cognition. A frozen
system never changes regime. A noisy one changes every step. A system with one
dominant regime spends its life there and visits the others by accident. A
system with as many regimes as timesteps has no regimes at all.

So the trajectory is clustered, and three quantities are read off the sequence
of labels: how long it stays (dwell), how predictable the next regime is given
this one (transition entropy, which has to be above zero and below its
ceiling), and how much of the life the most common regime takes. Metastability
is the middle of all three.

The number of regimes is chosen by silhouette score over a small range rather
than fixed, because fixing it decides the answer: ask for two and any
trajectory looks bistable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.subject.recording import Recording

__all__ = ["MetastabilityReport", "condition_alignment", "regimes"]

CANDIDATE_K: tuple[int, ...] = (2, 3, 4, 5, 6, 8)


def _kmeans(x: np.ndarray, k: int, *, seed: int, iterations: int = 60) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = x[rng.choice(x.shape[0], size=k, replace=False)]
    labels = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(iterations):
        distances = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new = distances.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for index in range(k):
            members = x[labels == index]
            if members.size:
                centres[index] = members.mean(axis=0)
    return labels


def _silhouette(x: np.ndarray, labels: np.ndarray, *, sample: int = 600, seed: int = 0) -> float:
    unique = np.unique(labels)
    if unique.size < 2:
        return -1.0
    rng = np.random.default_rng(seed)
    rows = rng.choice(x.shape[0], size=min(sample, x.shape[0]), replace=False)
    subset, sub_labels = x[rows], labels[rows]
    distances = np.sqrt(((subset[:, None, :] - subset[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for index in range(subset.shape[0]):
        own = sub_labels[index]
        same = distances[index][(sub_labels == own)]
        same = same[same > 0]
        if same.size == 0:
            continue
        a = same.mean()
        b = min(
            distances[index][sub_labels == other].mean()
            for other in unique
            if other != own and np.any(sub_labels == other)
        )
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores)) if scores else -1.0


def condition_alignment(labels: Any, conditions: Any) -> dict[str, float]:
    """How much of the regime sequence the condition labels already account for.

    A clustering of a life run under eight named conditions can find exactly
    those eight and call them regimes. What separates a regime from a relabelled
    condition is whether the regime says anything the condition does not: the
    entropy left in the regime once the condition is known. Zero means the
    regimes are a function of the conditions. The normalised mutual information
    says how far towards that the two are, and one is the rediscovery.
    """
    pairs = [(int(label), str(condition)) for label, condition in zip(labels, conditions, strict=False)]
    n = len(pairs)
    if n == 0:
        return {"condition_nmi": 0.0, "regime_given_condition_bits": 0.0, "conditions_spanning_regimes": 0.0}
    from collections import Counter

    regime_counts = Counter(label for label, _ in pairs)
    condition_counts = Counter(condition for _, condition in pairs)
    joint_counts = Counter(pairs)

    def entropy(counts: Counter) -> float:
        return float(-sum((c / n) * math.log2(c / n) for c in counts.values() if c))

    h_regime, h_condition, h_joint = entropy(regime_counts), entropy(condition_counts), entropy(joint_counts)
    mutual = h_regime + h_condition - h_joint
    nmi = mutual / math.sqrt(h_regime * h_condition) if h_regime > 0 and h_condition > 0 else 0.0
    spanning = sum(
        1 for condition in condition_counts
        if len({label for label, other in pairs if other == condition}) > 1
    )
    return {
        "condition_nmi": round(float(nmi), 4),
        "regime_given_condition_bits": round(float(h_joint - h_condition), 4),
        "conditions_spanning_regimes": float(spanning),
    }


@dataclass
class MetastabilityReport:
    regimes: int
    dwell_mean: float
    dwell_max: float
    transition_entropy: float
    entropy_ceiling: float
    top_share: float
    labels: tuple[int, ...]
    #: See `condition_alignment`. Empty when the recording carried no conditions.
    alignment: dict[str, float] | None = None

    @property
    def not_the_conditions(self) -> bool | None:
        """The regimes say something the condition labels do not. None when unmeasured."""
        if not self.alignment:
            return None
        return (
            self.alignment["regime_given_condition_bits"] > 0.0
            and self.alignment["condition_nmi"] < 1.0
        )

    @property
    def passes(self) -> bool:
        return (
            self.regimes > 1
            and 0.0 < self.transition_entropy < self.entropy_ceiling
            and 0.0 < self.top_share < 1.0
            and self.dwell_mean > 1.0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "regimes": self.regimes,
            "dwell_mean": round(self.dwell_mean, 3),
            "dwell_max": round(self.dwell_max, 3),
            "transition_entropy": round(self.transition_entropy, 4),
            "entropy_ceiling": round(self.entropy_ceiling, 4),
            "largest_regime_share": round(self.top_share, 4),
            "passes": self.passes,
            **(self.alignment or {}),
            "not_the_conditions": self.not_the_conditions,
        }


def regimes(recording: Recording, *, seed: int = 0) -> MetastabilityReport:
    live = recording.live_columns()
    block = recording.x[:, live]
    if block.shape[0] < 40 or block.shape[1] < 2:
        return MetastabilityReport(1, 0.0, 0.0, 0.0, 0.0, 1.0, ())
    spread = block.std(axis=0)
    block = (block - block.mean(axis=0)) / np.maximum(spread, 1e-9)

    best_k, best_labels, best_score = 1, np.zeros(block.shape[0], dtype=np.int64), -2.0
    for k in CANDIDATE_K:
        if k >= block.shape[0]:
            continue
        labels = _kmeans(block, k, seed=seed)
        if np.unique(labels).size < 2:
            continue
        score = _silhouette(block, labels, seed=seed)
        if score > best_score:
            best_k, best_labels, best_score = k, labels, score

    labels = best_labels
    counts = np.bincount(labels, minlength=best_k).astype(np.float64)
    share = counts / counts.sum()

    runs: list[int] = []
    current = 1
    for index in range(1, labels.size):
        if labels[index] == labels[index - 1]:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)

    joint = np.zeros((best_k, best_k), dtype=np.float64)
    for index in range(labels.size - 1):
        joint[labels[index], labels[index + 1]] += 1.0
    entropy = 0.0
    total = joint.sum()
    if total > 0:
        for row in range(best_k):
            row_total = joint[row].sum()
            if row_total <= 0:
                continue
            p_row = row_total / total
            probabilities = joint[row] / row_total
            probabilities = probabilities[probabilities > 0]
            entropy += p_row * float(-(probabilities * np.log2(probabilities)).sum())

    return MetastabilityReport(
        regimes=best_k,
        dwell_mean=float(np.mean(runs)),
        dwell_max=float(np.max(runs)),
        transition_entropy=entropy,
        entropy_ceiling=math.log2(best_k) if best_k > 1 else 0.0,
        top_share=float(share.max()),
        labels=tuple(int(v) for v in labels),
        alignment=(
            condition_alignment(labels, recording.conditions)
            if len(getattr(recording, "conditions", ()) or ()) == labels.size
            else None
        ),
    )
