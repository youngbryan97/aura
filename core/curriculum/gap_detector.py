"""Detect knowledge gaps from the prediction ledger.

A gap is a (belief, modality) cluster where Aura's recent predictions
are systematically wrong.  ``GapDetector.detect`` returns the worst
cluster — the one with the highest mean Brier — as the focus for the
next curriculum iteration.

We require a minimum number of resolved predictions per cluster
(``min_resolved``) so a single bad prediction doesn't trigger a full
learning loop.  Clusters tied at the worst Brier are broken by
recency: the most recently active cluster wins, on the theory that
recent failures are more relevant than ancient ones.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.runtime.prediction_ledger import PredictionLedger, PredictionRecord


@dataclass
class GapReport:
    cluster: Optional[str]
    belief: Optional[str]
    modality: Optional[str]
    n_resolved: int
    mean_brier: float
    accuracy: float
    last_seen_at: float = 0.0

    @property
    def has_gap(self) -> bool:
        return self.cluster is not None


class GapDetector:
    def __init__(
        self,
        ledger: PredictionLedger,
        *,
        min_resolved: int = 3,
        brier_threshold: float = 0.10,
    ):
        self.ledger = ledger
        self.min_resolved = int(min_resolved)
        self.brier_threshold = float(brier_threshold)

    def _cluster_key(self, record: PredictionRecord) -> str:
        return f"{record.modality}::{record.belief}"

    def detect(self, *, window: int = 200) -> GapReport:
        records = self.ledger.recent(limit=window, resolved=True)
        clusters: Dict[str, List[PredictionRecord]] = defaultdict(list)
        for r in records:
            clusters[self._cluster_key(r)].append(r)

        best: Optional[Tuple[str, float, float, int, float]] = None
        for cluster_id, items in clusters.items():
            if len(items) < self.min_resolved:
                continue
            briers = [r.brier for r in items if r.brier is not None]
            if not briers:
                continue
            mean_brier = sum(briers) / len(briers)
            if mean_brier < self.brier_threshold:
                continue
            correct = sum(1 for r in items if (r.brier or 1.0) <= 0.05)
            accuracy = correct / len(items)
            last_seen = max((r.resolved_at or r.created_at) for r in items)
            score = (mean_brier, last_seen)
            if (
                best is None
                or score > (best[1], best[4])
            ):
                best = (cluster_id, mean_brier, accuracy, len(items), last_seen)

        if best is None:
            return GapReport(
                cluster=None,
                belief=None,
                modality=None,
                n_resolved=0,
                mean_brier=0.0,
                accuracy=1.0,
                last_seen_at=0.0,
            )

        cluster_id, mean_brier, accuracy, count, last_seen = best
        modality, _, belief = cluster_id.partition("::")
        return GapReport(
            cluster=cluster_id,
            belief=belief,
            modality=modality,
            n_resolved=count,
            mean_brier=mean_brier,
            accuracy=accuracy,
            last_seen_at=last_seen,
        )
