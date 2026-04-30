"""Empirical calibration for autonomous repair decisions.

LLM confidence is not enough for safe self-repair.  This module records
predicted correctness versus observed outcomes by bug class, mutation tier, and
module family, then returns a calibrated probability used by promotion gates.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CalibrationBucket:
    bug_class: str
    risk_tier: str
    module_family: str
    attempts: int
    observed_success_rate: float
    mean_predicted: float
    brier_score: float
    expected_calibration_error: float

    @property
    def autonomous_safe(self) -> bool:
        return self.attempts >= 8 and self.observed_success_rate >= 0.90 and self.expected_calibration_error <= 0.12

    def to_dict(self) -> dict[str, Any]:
        return {
            "bug_class": self.bug_class,
            "risk_tier": self.risk_tier,
            "module_family": self.module_family,
            "attempts": self.attempts,
            "observed_success_rate": round(self.observed_success_rate, 4),
            "mean_predicted": round(self.mean_predicted, 4),
            "brier_score": round(self.brier_score, 4),
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "autonomous_safe": self.autonomous_safe,
        }


class RepairCalibrationStore:
    """Persistent repair reliability tracker."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".aura" / "data" / "selfmod" / "repair_calibration.sqlite3"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS repair_calibration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bug_class TEXT NOT NULL,
                    risk_tier TEXT NOT NULL,
                    module_family TEXT NOT NULL,
                    predicted REAL NOT NULL,
                    outcome INTEGER NOT NULL,
                    patch_id TEXT,
                    created_at REAL NOT NULL
                )
                """
            )

    def record(
        self,
        *,
        bug_class: str,
        risk_tier: str,
        module_family: str,
        predicted: float,
        outcome: bool,
        patch_id: str = "",
    ) -> None:
        predicted = max(0.0, min(1.0, float(predicted)))
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO repair_calibration
                    (bug_class, risk_tier, module_family, predicted, outcome, patch_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (bug_class, risk_tier, module_family, predicted, int(bool(outcome)), patch_id, time.time()),
            )

    def bucket(self, *, bug_class: str, risk_tier: str, module_family: str) -> CalibrationBucket:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT predicted, outcome FROM repair_calibration
                WHERE bug_class = ? AND risk_tier = ? AND module_family = ?
                """,
                (bug_class, risk_tier, module_family),
            ).fetchall()
        if not rows:
            return CalibrationBucket(bug_class, risk_tier, module_family, 0, 0.0, 0.5, 0.25, 1.0)
        preds = [float(row[0]) for row in rows]
        outs = [float(row[1]) for row in rows]
        attempts = len(rows)
        observed = sum(outs) / attempts
        mean_predicted = sum(preds) / attempts
        brier = sum((p - o) ** 2 for p, o in zip(preds, outs)) / attempts
        ece = abs(mean_predicted - observed)
        return CalibrationBucket(bug_class, risk_tier, module_family, attempts, observed, mean_predicted, brier, ece)

    def calibrated_probability(
        self,
        *,
        bug_class: str,
        risk_tier: str,
        module_family: str,
        model_confidence: float,
    ) -> float:
        bucket = self.bucket(bug_class=bug_class, risk_tier=risk_tier, module_family=module_family)
        raw = max(0.0, min(1.0, float(model_confidence)))
        if bucket.attempts < 4:
            return min(raw, 0.55)
        return max(0.0, min(1.0, (raw * 0.35) + (bucket.observed_success_rate * 0.65) - bucket.expected_calibration_error))

    def should_escalate(
        self,
        *,
        bug_class: str,
        risk_tier: str,
        module_family: str,
        model_confidence: float,
        threshold: float = 0.80,
    ) -> bool:
        probability = self.calibrated_probability(
            bug_class=bug_class,
            risk_tier=risk_tier,
            module_family=module_family,
            model_confidence=model_confidence,
        )
        bucket = self.bucket(bug_class=bug_class, risk_tier=risk_tier, module_family=module_family)
        return probability < threshold or (risk_tier in {"tier2_propose_only", "tier3_sealed"} and not bucket.autonomous_safe)

    def export_summary(self) -> dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT bug_class, risk_tier, module_family, COUNT(*), AVG(outcome), AVG(predicted)
                FROM repair_calibration
                GROUP BY bug_class, risk_tier, module_family
                """
            ).fetchall()
        return {
            "generated_at": time.time(),
            "buckets": [
                {
                    "bug_class": row[0],
                    "risk_tier": row[1],
                    "module_family": row[2],
                    "attempts": int(row[3]),
                    "observed_success_rate": float(row[4] or 0.0),
                    "mean_predicted": float(row[5] or 0.0),
                }
                for row in rows
            ],
        }


_instance: RepairCalibrationStore | None = None


def get_repair_calibration() -> RepairCalibrationStore:
    global _instance
    if _instance is None:
        _instance = RepairCalibrationStore()
    return _instance


__all__ = ["CalibrationBucket", "RepairCalibrationStore", "get_repair_calibration"]
