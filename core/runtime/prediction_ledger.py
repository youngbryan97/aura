"""Prediction ledger — durable record of belief -> action -> outcome.

Aura claims to have beliefs and to act on them.  Without a forensic
record of what she predicted, what happened, and how she updated, that
claim is unverifiable.  This module provides the ledger that turns it
into a measurement.

For every prediction the system makes:

    register()      -> create a row with prior probability and the
                       expected observation, before the action is taken
    resolve()       -> attach the observed outcome, compute Brier error,
                       and record the posterior + any policy update

From the resolved rows we derive:

    score_brier()   -> mean per-prediction Brier loss (lower is better)
    calibration()   -> reliability diagram (binned predicted vs. actual
                       frequency); calibration_error returns the
                       expected calibration error (ECE)

The ledger is SQLite-backed for crash-durability and queryability, and
schema-versioned so future revisions can migrate without losing rows.
The "expected" and "observed" columns are JSON-encoded so callers may
record arbitrary structured outcomes; the per-class Brier helper covers
both binary (single ``prior_prob``) and categorical (``prior_dist``)
predictions.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCHEMA_VERSION = 1


@dataclass
class PredictionRecord:
    prediction_id: str
    created_at: float
    resolved_at: Optional[float]
    belief: str
    modality: str
    action: str
    expected: Any
    observed: Any
    prior_prob: Optional[float]
    posterior_prob: Optional[float]
    prior_dist: Optional[Dict[str, float]]
    posterior_dist: Optional[Dict[str, float]]
    brier: Optional[float]
    error: Optional[float]
    intervention_id: Optional[str]
    agent_state: Dict[str, Any] = field(default_factory=dict)
    policy_change: Optional[str] = None
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "belief": self.belief,
            "modality": self.modality,
            "action": self.action,
            "expected": self.expected,
            "observed": self.observed,
            "prior_prob": self.prior_prob,
            "posterior_prob": self.posterior_prob,
            "prior_dist": self.prior_dist,
            "posterior_dist": self.posterior_dist,
            "brier": self.brier,
            "error": self.error,
            "intervention_id": self.intervention_id,
            "agent_state": self.agent_state,
            "policy_change": self.policy_change,
            "resolved": self.resolved,
        }


class PredictionLedgerError(RuntimeError):
    pass


def _new_prediction_id() -> str:
    return f"pred-{uuid.uuid4()}"


def _binary_brier(prior_prob: float, observed_truth: bool) -> float:
    o = 1.0 if observed_truth else 0.0
    return (float(prior_prob) - o) ** 2


def _categorical_brier(prior_dist: Dict[str, float], observed_class: str) -> float:
    """Multi-class Brier score in the standard formulation:

        BS = sum_c (p_c - o_c)^2

    where ``o_c`` is 1 for the observed class and 0 elsewhere.
    """
    total = 0.0
    seen_observed = False
    for cls, p in prior_dist.items():
        o = 1.0 if cls == observed_class else 0.0
        if cls == observed_class:
            seen_observed = True
        total += (float(p) - o) ** 2
    if not seen_observed:
        # The observed class was missing from the prior distribution.
        # Treat its prior as zero, which contributes (0 - 1)^2 = 1.
        total += 1.0
    return total


class PredictionLedger:
    """SQLite-backed durable ledger for predictions and outcomes."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = (
            Path(db_path)
            if db_path is not None
            else Path.home() / ".aura" / "data" / "prediction_ledger.db"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    # ------------------------------------------------------------------
    # schema
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id    TEXT PRIMARY KEY,
                    created_at       REAL NOT NULL,
                    resolved_at      REAL,
                    belief           TEXT NOT NULL,
                    modality         TEXT NOT NULL,
                    action           TEXT NOT NULL,
                    expected_json    TEXT NOT NULL,
                    observed_json    TEXT,
                    prior_prob       REAL,
                    posterior_prob   REAL,
                    prior_dist_json  TEXT,
                    posterior_dist_json TEXT,
                    brier            REAL,
                    error            REAL,
                    intervention_id  TEXT,
                    agent_state_json TEXT NOT NULL DEFAULT '{}',
                    policy_change    TEXT,
                    resolved         INTEGER NOT NULL DEFAULT 0,
                    schema_version   INTEGER NOT NULL DEFAULT 1
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_resolved ON predictions(resolved);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_intervention ON predictions(intervention_id);"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    component TEXT PRIMARY KEY,
                    version   INTEGER NOT NULL
                );
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO schema_version(component, version) VALUES (?, ?);",
                ("prediction_ledger", SCHEMA_VERSION),
            )

    # ------------------------------------------------------------------
    # write path
    # ------------------------------------------------------------------
    def register(
        self,
        *,
        belief: str,
        modality: str,
        action: str,
        expected: Any,
        prior_prob: Optional[float] = None,
        prior_dist: Optional[Dict[str, float]] = None,
        intervention_id: Optional[str] = None,
        agent_state: Optional[Dict[str, Any]] = None,
        prediction_id: Optional[str] = None,
        created_at: Optional[float] = None,
    ) -> str:
        """Record a prediction before its outcome is known.

        Either ``prior_prob`` (binary belief) or ``prior_dist``
        (categorical belief over class labels) must be supplied; the
        ledger refuses ambiguous rows.
        """
        if prior_prob is None and prior_dist is None:
            raise PredictionLedgerError(
                "register() requires prior_prob or prior_dist"
            )
        if prior_prob is not None and not (0.0 <= float(prior_prob) <= 1.0):
            raise PredictionLedgerError("prior_prob must be in [0, 1]")
        if prior_dist is not None:
            total = sum(float(v) for v in prior_dist.values())
            if not (0.99 <= total <= 1.01):
                raise PredictionLedgerError(
                    f"prior_dist must sum to ~1.0, got {total:.4f}"
                )
        pred_id = prediction_id or _new_prediction_id()
        ts = float(created_at if created_at is not None else time.time())
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO predictions(
                    prediction_id, created_at, belief, modality, action,
                    expected_json, prior_prob, prior_dist_json,
                    intervention_id, agent_state_json, resolved,
                    schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?);
                """,
                (
                    pred_id,
                    ts,
                    belief,
                    modality,
                    action,
                    json.dumps(expected, sort_keys=True, default=str),
                    None if prior_prob is None else float(prior_prob),
                    None if prior_dist is None else json.dumps(prior_dist, sort_keys=True),
                    intervention_id,
                    json.dumps(agent_state or {}, sort_keys=True, default=str),
                    SCHEMA_VERSION,
                ),
            )
        return pred_id

    def resolve(
        self,
        prediction_id: str,
        *,
        observed: Any,
        observed_truth: Optional[bool] = None,
        observed_class: Optional[str] = None,
        posterior_prob: Optional[float] = None,
        posterior_dist: Optional[Dict[str, float]] = None,
        policy_change: Optional[str] = None,
        resolved_at: Optional[float] = None,
    ) -> PredictionRecord:
        """Attach the observed outcome and compute Brier error.

        For binary predictions, supply ``observed_truth`` (True/False).
        For categorical predictions, supply ``observed_class`` (a key
        present in the registered ``prior_dist``).
        """
        ts = float(resolved_at if resolved_at is not None else time.time())
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE prediction_id = ?",
                (prediction_id,),
            ).fetchone()
            if row is None:
                raise PredictionLedgerError(
                    f"unknown prediction_id: {prediction_id}"
                )
            if int(row["resolved"]) != 0:
                raise PredictionLedgerError(
                    f"prediction already resolved: {prediction_id}"
                )

            prior_prob = row["prior_prob"]
            prior_dist_json = row["prior_dist_json"]
            prior_dist = json.loads(prior_dist_json) if prior_dist_json else None

            if prior_prob is not None and prior_dist is None:
                if observed_truth is None:
                    raise PredictionLedgerError(
                        "binary prediction requires observed_truth=True/False"
                    )
                brier = _binary_brier(prior_prob, observed_truth)
                error = abs(float(prior_prob) - (1.0 if observed_truth else 0.0))
            elif prior_dist is not None:
                if observed_class is None:
                    raise PredictionLedgerError(
                        "categorical prediction requires observed_class"
                    )
                brier = _categorical_brier(prior_dist, observed_class)
                expected_p = float(prior_dist.get(observed_class, 0.0))
                error = 1.0 - expected_p
            else:
                raise PredictionLedgerError(
                    f"prediction {prediction_id} has neither prior_prob nor prior_dist"
                )

            conn.execute(
                """
                UPDATE predictions SET
                    resolved = 1,
                    resolved_at = ?,
                    observed_json = ?,
                    posterior_prob = ?,
                    posterior_dist_json = ?,
                    brier = ?,
                    error = ?,
                    policy_change = ?
                WHERE prediction_id = ?;
                """,
                (
                    ts,
                    json.dumps(observed, sort_keys=True, default=str),
                    None if posterior_prob is None else float(posterior_prob),
                    None
                    if posterior_dist is None
                    else json.dumps(posterior_dist, sort_keys=True),
                    float(brier),
                    float(error),
                    policy_change,
                    prediction_id,
                ),
            )

        return self.get(prediction_id)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # read path
    # ------------------------------------------------------------------
    def get(self, prediction_id: str) -> Optional[PredictionRecord]:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE prediction_id = ?",
                (prediction_id,),
            ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def recent(self, limit: int = 50, *, resolved: Optional[bool] = None) -> List[PredictionRecord]:
        if limit <= 0:
            return []
        clause = ""
        params: Tuple[Any, ...] = ()
        if resolved is True:
            clause = " WHERE resolved = 1"
        elif resolved is False:
            clause = " WHERE resolved = 0"
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT * FROM predictions{clause} ORDER BY created_at DESC LIMIT ?",
                (*params, int(limit)),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def _row_to_record(self, row: sqlite3.Row) -> PredictionRecord:
        return PredictionRecord(
            prediction_id=row["prediction_id"],
            created_at=float(row["created_at"]),
            resolved_at=float(row["resolved_at"]) if row["resolved_at"] is not None else None,
            belief=row["belief"],
            modality=row["modality"],
            action=row["action"],
            expected=json.loads(row["expected_json"]) if row["expected_json"] else None,
            observed=json.loads(row["observed_json"]) if row["observed_json"] else None,
            prior_prob=row["prior_prob"],
            posterior_prob=row["posterior_prob"],
            prior_dist=json.loads(row["prior_dist_json"]) if row["prior_dist_json"] else None,
            posterior_dist=(
                json.loads(row["posterior_dist_json"])
                if row["posterior_dist_json"]
                else None
            ),
            brier=row["brier"],
            error=row["error"],
            intervention_id=row["intervention_id"],
            agent_state=json.loads(row["agent_state_json"]) if row["agent_state_json"] else {},
            policy_change=row["policy_change"],
            resolved=bool(row["resolved"]),
        )

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    def score_brier(
        self,
        *,
        since: Optional[float] = None,
        until: Optional[float] = None,
        modality: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mean Brier loss over resolved predictions in a window.

        Returns a dict with ``count`` and ``mean_brier``.  When no rows
        match, ``mean_brier`` is None.
        """
        clauses = ["resolved = 1", "brier IS NOT NULL"]
        params: List[Any] = []
        if since is not None:
            clauses.append("resolved_at >= ?")
            params.append(float(since))
        if until is not None:
            clauses.append("resolved_at <= ?")
            params.append(float(until))
        if modality is not None:
            clauses.append("modality = ?")
            params.append(modality)
        where = " AND ".join(clauses)
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS n, AVG(brier) AS mean_brier FROM predictions WHERE {where};",
                params,
            ).fetchone()
        n = int(row["n"]) if row["n"] is not None else 0
        mean = float(row["mean_brier"]) if row["mean_brier"] is not None else None
        return {"count": n, "mean_brier": mean}

    def calibration(
        self,
        *,
        bins: int = 10,
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Reliability diagram for *binary* predictions (prior_prob set).

        Returns a dict with ``bins`` (list of {lower, upper, count,
        mean_predicted, actual_rate}) and ``ece`` (expected calibration
        error: weighted absolute gap between predicted and actual).
        """
        if bins <= 0:
            raise PredictionLedgerError("bins must be positive")
        clauses = ["resolved = 1", "prior_prob IS NOT NULL"]
        params: List[Any] = []
        if since is not None:
            clauses.append("resolved_at >= ?")
            params.append(float(since))
        if until is not None:
            clauses.append("resolved_at <= ?")
            params.append(float(until))
        where = " AND ".join(clauses)
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                f"""
                SELECT prior_prob, observed_json
                FROM predictions
                WHERE {where};
                """,
                params,
            ).fetchall()

        bin_data: List[Dict[str, Any]] = []
        for i in range(bins):
            lo = i / bins
            hi = (i + 1) / bins
            bin_data.append(
                {
                    "lower": lo,
                    "upper": hi,
                    "count": 0,
                    "sum_predicted": 0.0,
                    "sum_actual": 0.0,
                }
            )

        total = 0
        for row in rows:
            p = float(row["prior_prob"])
            observed_raw = row["observed_json"]
            try:
                observed = json.loads(observed_raw) if observed_raw else None
            except json.JSONDecodeError:
                continue
            truth = self._coerce_truth(observed)
            if truth is None:
                continue
            idx = min(bins - 1, max(0, int(p * bins)))
            bin_data[idx]["count"] += 1
            bin_data[idx]["sum_predicted"] += p
            bin_data[idx]["sum_actual"] += 1.0 if truth else 0.0
            total += 1

        ece = 0.0
        for b in bin_data:
            if b["count"] == 0:
                b["mean_predicted"] = None
                b["actual_rate"] = None
                continue
            b["mean_predicted"] = b["sum_predicted"] / b["count"]
            b["actual_rate"] = b["sum_actual"] / b["count"]
            ece += (b["count"] / total) * abs(b["mean_predicted"] - b["actual_rate"])
            del b["sum_predicted"]
            del b["sum_actual"]

        # Strip helper keys for empty bins too.
        for b in bin_data:
            b.pop("sum_predicted", None)
            b.pop("sum_actual", None)

        return {"bins": bin_data, "ece": ece, "total": total}

    @staticmethod
    def _coerce_truth(observed: Any) -> Optional[bool]:
        """Best-effort extraction of a boolean truth value from the observed payload."""
        if isinstance(observed, bool):
            return observed
        if isinstance(observed, dict):
            for key in ("truth", "outcome", "observed_truth", "value"):
                if key in observed:
                    val = observed[key]
                    if isinstance(val, bool):
                        return val
                    if isinstance(val, (int, float)):
                        return bool(val)
                    if isinstance(val, str):
                        if val.lower() in {"true", "1", "yes", "y"}:
                            return True
                        if val.lower() in {"false", "0", "no", "n"}:
                            return False
        if isinstance(observed, (int, float)):
            return bool(observed)
        return None

    # ------------------------------------------------------------------
    # housekeeping
    # ------------------------------------------------------------------
    def count(self) -> int:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM predictions;").fetchone()
        return int(row["n"]) if row is not None else 0

    def iter_unresolved(self) -> Iterable[PredictionRecord]:
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE resolved = 0 ORDER BY created_at ASC;"
            ).fetchall()
        for r in rows:
            yield self._row_to_record(r)


_global_ledger: Optional[PredictionLedger] = None
_singleton_lock = threading.RLock()


def get_prediction_ledger(db_path: Optional[Path] = None) -> PredictionLedger:
    global _global_ledger
    with _singleton_lock:
        if _global_ledger is None:
            _global_ledger = PredictionLedger(db_path)
        return _global_ledger


def reset_prediction_ledger() -> None:
    global _global_ledger
    with _singleton_lock:
        _global_ledger = None
