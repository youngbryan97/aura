"""Improvement-curve recorder.

After every iteration the loop snapshots Brier loss and pass-rate over
the most recent N predictions.  ``trend`` reports whether the system
is improving, plateauing, or regressing — the only signal a long-run
operator needs to know if the loop is doing real work.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import List, Optional

from core.runtime.prediction_ledger import PredictionLedger


@dataclass
class ImprovementSnapshot:
    iteration: int
    n_resolved: int
    mean_brier: Optional[float]
    accuracy: float


class ImprovementRecorder:
    def __init__(self, ledger: PredictionLedger, *, window: int = 50):
        self.ledger = ledger
        self.window = int(window)
        self.snapshots: List[ImprovementSnapshot] = []

    def snapshot(self, iteration: int) -> ImprovementSnapshot:
        records = self.ledger.recent(limit=self.window, resolved=True)
        if not records:
            snap = ImprovementSnapshot(iteration=iteration, n_resolved=0, mean_brier=None, accuracy=0.0)
        else:
            briers = [r.brier for r in records if r.brier is not None]
            mean_brier = statistics.fmean(briers) if briers else None
            correct = sum(1 for r in records if (r.brier or 1.0) <= 0.05)
            snap = ImprovementSnapshot(
                iteration=iteration,
                n_resolved=len(records),
                mean_brier=mean_brier,
                accuracy=correct / len(records),
            )
        self.snapshots.append(snap)
        return snap

    def trend(self) -> str:
        """Return 'improving' | 'plateau' | 'regressing' | 'unknown'."""
        if len(self.snapshots) < 2:
            return "unknown"
        first = self.snapshots[0].mean_brier
        last = self.snapshots[-1].mean_brier
        if first is None or last is None:
            return "unknown"
        if last < first - 1e-6:
            return "improving"
        if last > first + 1e-6:
            return "regressing"
        return "plateau"

    def to_dict_list(self) -> list:
        return [
            {
                "iteration": s.iteration,
                "n_resolved": s.n_resolved,
                "mean_brier": s.mean_brier,
                "accuracy": s.accuracy,
            }
            for s in self.snapshots
        ]
