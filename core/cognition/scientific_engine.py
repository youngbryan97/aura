"""Scientific engine — hypothesis → experiment → belief update, as a real causal loop.

The critique asked for a scientific engine that forms hypotheses, runs experiments, and
updates beliefs from the result — not a prompt that *talks about* doing science. The way to
make it causal rather than an island is to reuse machinery that already exists: an
experiment here is literally an Outcome-Ledger receipt. Forming a hypothesis commits an
*expected* observable; running the experiment opens a receipt with that expectation;
observing the result resolves the receipt (so the ledger computes expected-vs-observed and
assigns credit), and the same observation drives a Bayesian-flavored confidence update on
the belief.

    h = engine.form_hypothesis("retrying flaky tool fixes it", predicted="success_rate",
                               expected=0.8, prior_confidence=0.5)
    engine.run_experiment(h)              # opens an OutcomeLedger receipt
    engine.observe(h, observed=0.9)       # resolves the receipt + revises the belief up

Hypotheses persist in SQLite, so an experiment opened in one session can be observed and
fold into beliefs in another — a genuinely long-horizon inquiry loop. Confidence updates
are best-effort pushed to whatever belief sink the runtime exposes, so the conclusions are
visible to the rest of cognition, not trapped here.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.config import config
from core.runtime.errors import record_degradation
from core.runtime.sqlite_support import connecting

logger = logging.getLogger("Cognition.ScientificEngine")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class Hypothesis:
    hypothesis_id: str
    claim: str
    predicted_observable: str
    expected: float
    confidence: float
    status: str = "open"            # open | supported | refuted | inconclusive
    trials: int = 0
    supports: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    receipt_id: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "claim": self.claim,
            "predicted_observable": self.predicted_observable,
            "expected": self.expected,
            "confidence": self.confidence,
            "status": self.status,
            "trials": self.trials,
            "supports": self.supports,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "receipt_id": self.receipt_id,
            "evidence": self.evidence,
        }


class ScientificEngine:
    """Forms hypotheses, runs experiments (as ledger receipts), and revises beliefs."""

    def __init__(
        self,
        db_path: str | None = None,
        *,
        learning_rate: float = 0.4,
        support_threshold: float = 0.15,
    ) -> None:
        self._db_path = db_path or str(config.paths.home_dir / "data/scientific_engine.db")
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._lr = learning_rate
        self._support_threshold = support_threshold  # |error| below this counts as support
        self._cache: dict[str, Hypothesis] = {}
        self._init_schema()
        self._load()

    # ── persistence ──────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_schema(self) -> None:
        try:
            with connecting(self._connect()) as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS hypotheses (
                        hypothesis_id TEXT PRIMARY KEY,
                        claim TEXT NOT NULL,
                        predicted_observable TEXT NOT NULL,
                        expected REAL NOT NULL,
                        confidence REAL NOT NULL,
                        status TEXT NOT NULL,
                        trials INTEGER NOT NULL,
                        supports INTEGER NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        receipt_id TEXT,
                        evidence_json TEXT
                    )"""
                )
                conn.commit()
        except (sqlite3.Error, OSError) as e:
            record_degradation("scientific_engine", e)

    def _persist(self, h: Hypothesis) -> None:
        try:
            with connecting(self._connect()) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO hypotheses
                       (hypothesis_id, claim, predicted_observable, expected, confidence, status,
                        trials, supports, created_at, updated_at, receipt_id, evidence_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        h.hypothesis_id, h.claim, h.predicted_observable, h.expected, h.confidence,
                        h.status, h.trials, h.supports, h.created_at, h.updated_at, h.receipt_id,
                        json.dumps(h.evidence[-50:]),
                    ),
                )
                conn.commit()
        except (sqlite3.Error, OSError) as e:
            record_degradation("scientific_engine", e)

    def _load(self) -> None:
        try:
            with connecting(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT hypothesis_id, claim, predicted_observable, expected, confidence, status, "
                    "trials, supports, created_at, updated_at, receipt_id, evidence_json FROM hypotheses"
                ).fetchall()
            for r in rows:
                self._cache[r[0]] = Hypothesis(
                    hypothesis_id=r[0], claim=r[1], predicted_observable=r[2], expected=r[3],
                    confidence=r[4], status=r[5], trials=r[6], supports=r[7], created_at=r[8],
                    updated_at=r[9], receipt_id=r[10], evidence=json.loads(r[11] or "[]"),
                )
        except (sqlite3.Error, OSError, ValueError) as e:
            record_degradation("scientific_engine", e)

    # ── the inquiry loop ──────────────────────────────────────────────────

    def form_hypothesis(
        self,
        claim: str,
        *,
        predicted_observable: str,
        expected: float,
        prior_confidence: float = 0.5,
    ) -> str:
        """State a falsifiable claim: an observable and its expected value."""
        h = Hypothesis(
            hypothesis_id=f"hyp-{uuid.uuid4().hex[:10]}",
            claim=claim,
            predicted_observable=predicted_observable,
            expected=_clamp(float(expected)),
            confidence=_clamp(float(prior_confidence)),
        )
        with self._lock:
            self._cache[h.hypothesis_id] = h
            self._persist(h)
        return h.hypothesis_id

    def run_experiment(self, hypothesis_id: str, *, horizon_s: float = 3600.0) -> str | None:
        """Open an Outcome-Ledger receipt carrying the hypothesis's prediction.

        The experiment *is* a receipt: the ledger will compute expected-vs-observed and
        assign credit to this hypothesis when it's resolved.
        """
        with self._lock:
            h = self._cache.get(hypothesis_id)
            if h is None:
                return None
        try:
            from core.cognition.outcome_ledger import CreditSource, get_outcome_ledger
            receipt_id = get_outcome_ledger().open(
                action=f"experiment:{h.claim}",
                expected=h.expected,
                sources=[CreditSource("hypothesis", h.hypothesis_id, 1.0)],
                category="experiment",
                horizon_s=horizon_s,
                context={"predicted_observable": h.predicted_observable},
            )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as e:
            record_degradation("scientific_engine", e, severity="debug",
                               action="ran experiment without ledger receipt")
            receipt_id = None
        with self._lock:
            h.receipt_id = receipt_id
            h.updated_at = time.time()
            self._persist(h)
        return receipt_id

    def observe(self, hypothesis_id: str, observed: float, *, note: str = "") -> Hypothesis | None:
        """Fold an observation into the hypothesis: resolve its receipt + revise the belief.

        Confidence moves toward support when |observed - expected| is small and away when it
        is large, by a learning-rate-scaled signal in [-1, 1]. After enough evidence the
        status settles to supported / refuted.
        """
        observed = _clamp(float(observed))
        with self._lock:
            h = self._cache.get(hypothesis_id)
            if h is None:
                return None
            error = abs(observed - h.expected)
            is_support = error <= self._support_threshold
            # signal: +1 perfect prediction, -1 maximal miss
            signal = 1.0 - 2.0 * error
            old_conf = h.confidence
            h.confidence = _clamp(h.confidence + self._lr * signal * (1.0 - h.confidence if signal > 0 else h.confidence))
            h.trials += 1
            if is_support:
                h.supports += 1
            h.evidence.append({
                "observed": observed, "expected": h.expected, "error": round(error, 4),
                "signal": round(signal, 4), "conf_before": round(old_conf, 4),
                "conf_after": round(h.confidence, 4), "at": time.time(), "note": note,
            })
            # Settle status once there is enough evidence.
            if h.trials >= 3:
                rate = h.supports / h.trials
                if rate >= 0.66 and h.confidence >= 0.6:
                    h.status = "supported"
                elif rate <= 0.34 and h.confidence <= 0.4:
                    h.status = "refuted"
                else:
                    h.status = "inconclusive"
            h.updated_at = time.time()
            receipt_id = h.receipt_id
            self._persist(h)

        # Resolve the experiment receipt (outside the lock — calls other subsystems).
        if receipt_id:
            try:
                from core.cognition.outcome_ledger import get_outcome_ledger
                get_outcome_ledger().resolve(receipt_id, observed, note=note or "scientific observation")
            except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as e:
                record_degradation("scientific_engine", e, severity="debug",
                                   action="observed result without resolving ledger receipt")

        self._publish_belief(h)
        return h

    def _publish_belief(self, h: Hypothesis) -> None:
        """Best-effort push of the revised belief to whatever belief sink the runtime exposes."""
        try:
            from core.runtime.service_registry import get_runtime_service

            ws = get_runtime_service("world_state", default=None)
            if ws is not None and hasattr(ws, "set_belief"):
                ws.set_belief(
                    f"hypothesis:{h.claim}",
                    {"status": h.status, "expected": h.expected},
                    confidence=h.confidence,
                    source="scientific_engine",
                )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as e:
            record_degradation("scientific_engine", e, severity="debug")

    # ── readout ──────────────────────────────────────────────────────────

    def get(self, hypothesis_id: str) -> Hypothesis | None:
        with self._lock:
            return self._cache.get(hypothesis_id)

    def belief(self, hypothesis_id: str) -> float | None:
        h = self.get(hypothesis_id)
        return None if h is None else h.confidence

    def supported(self) -> list[dict[str, Any]]:
        with self._lock:
            return [h.as_dict() for h in self._cache.values() if h.status == "supported"]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            by_status: dict[str, int] = {}
            for h in self._cache.values():
                by_status[h.status] = by_status.get(h.status, 0) + 1
            return {
                "total": len(self._cache),
                "by_status": by_status,
                "db_path": self._db_path,
            }


_engine: ScientificEngine | None = None
_engine_lock = threading.Lock()


def get_scientific_engine() -> ScientificEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = ScientificEngine()
    return _engine
