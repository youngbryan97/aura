"""Outcome Ledger — long-horizon credit assignment over delayed receipts.

OutcomeLearner already records outcomes, but it records them *at action time*, with the
result already known. Real credit assignment over a long horizon needs the opposite shape:
when the agent acts, it commits an *expectation* and notes which policy / memory / tool
contributed; the true outcome may not be observable until much later (a goal completes, a
reply lands, a tool's effect is verified). Only then can you compare expected vs observed,
measure the delay, and push credit back to the sources that earned it.

This ledger models exactly that lifecycle:

    receipt_id = ledger.open(action, expected, sources=[...])     # commit an expectation
    ...                                                            # (seconds → sessions later)
    ledger.resolve(receipt_id, observed=0.9)                      # the real outcome arrives

On resolve it computes ``delay``, ``prediction_error = observed - expected`` and a
success flag, then distributes credit to each contributing source (weighted by its share)
into the shared CreditAssignmentSystem, and records the outcome in the OutcomeLearner.
Receipts persist in SQLite (WAL), so an action opened in one session can be resolved — or
swept past its horizon — in another. That persistence is what makes the horizon genuinely
long instead of in-process-only.

A swept receipt keeps the accountability convention (observed=0, so sources that promised
an outcome nobody checked lose standing) while recording ``observation="unobserved"``, so
nothing downstream mistakes an assumed zero for a measurement. Statistics — the ledger's
own calibration included — use ``is_evidence``; credit assignment deliberately does not.

Sources are the seam the critique asked for: credit flows back to *policy*, *memory*, and
*tool* references, so the systems that produced a good (or bad) action are the ones whose
standing moves.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.config import config
from core.runtime.errors import record_degradation
from core.runtime.sqlite_support import connecting

logger = logging.getLogger("Cognition.OutcomeLedger")


#: Identical (category, action, context) opens inside this window fold into the still-
#: pending original instead of creating another row. Five minutes is long
#: enough to absorb a stuck loop and short enough that genuinely repeated work
#: an hour apart is still recorded separately.
_DEFAULT_COLLAPSE_WINDOW_S = 300.0


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class CreditSource:
    """A contributor to an action, to receive credit when the outcome lands."""

    kind: str          # "policy" | "memory" | "tool" | "plan" | "strategy" | ...
    ref: str           # identifier: policy name, memory id, tool name, ...
    weight: float = 1.0  # relative contribution share (normalized at resolve time)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "ref": self.ref, "weight": self.weight}


@dataclass
class OutcomeReceipt:
    receipt_id: str
    action: str
    category: str
    expected: float
    sources: list[CreditSource]
    opened_at: float
    horizon_s: float
    context: dict[str, Any] = field(default_factory=dict)
    observed: float | None = None
    resolved_at: float | None = None
    status: str = "pending"          # pending | resolved | expired
    prediction_error: float | None = None
    #: How ``observed`` was arrived at. ``"measured"`` means somebody actually
    #: looked; ``"unobserved"`` means the horizon passed and the zero is an
    #: accountability convention, not a fact about the world. Consumers doing
    #: statistics must exclude the latter; consumers doing credit assignment
    #: deliberately do not.
    observation: str = "measured"
    #: How many identical opens folded into this receipt. One stuck reflex is
    #: one fact with a weight, not a million rows.
    repeat_count: int = 1

    @property
    def delay(self) -> float | None:
        return None if self.resolved_at is None else self.resolved_at - self.opened_at

    def as_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "action": self.action,
            "category": self.category,
            "expected": self.expected,
            "sources": [s.as_dict() for s in self.sources],
            "opened_at": self.opened_at,
            "horizon_s": self.horizon_s,
            "context": self.context,
            "observed": self.observed,
            "resolved_at": self.resolved_at,
            "status": self.status,
            "prediction_error": self.prediction_error,
            "observation": self.observation,
            "repeat_count": self.repeat_count,
            "delay": self.delay,
        }

    @property
    def is_evidence(self) -> bool:
        """True when this receipt may be used as a training label or a statistic."""
        return self.observation == "measured" and self.observed is not None


class OutcomeLedger:
    """Delayed-receipt credit assignment with persistence and expectation calibration."""

    MAX_PENDING_LOAD = 5000

    def __init__(self, db_path: str | None = None, default_horizon_s: float = 3600.0) -> None:
        self._db_path = db_path or str(config.paths.home_dir / "data/outcome_ledger.db")
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._default_horizon = default_horizon_s
        self._pending: dict[str, OutcomeReceipt] = {}
        self._calib_err_sum = 0.0
        self._calib_n = 0
        self._unobserved_n = 0
        self._collapsed_opens = 0
        self._collapse_window = _DEFAULT_COLLAPSE_WINDOW_S
        self._open_index: dict[tuple[str, str, str], tuple[str, float]] = {}
        self._resolution_observers: list[Callable[[OutcomeReceipt], None]] = []
        self._pending_db_count = 0
        self._startup_expired_count = 0
        self._pending_load_truncated = False
        self._init_schema()
        self._load_pending()

    # ── persistence ──────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_schema(self) -> None:
        try:
            with connecting(self._connect()) as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS outcome_receipts (
                        receipt_id TEXT PRIMARY KEY,
                        action TEXT NOT NULL,
                        category TEXT NOT NULL,
                        expected REAL NOT NULL,
                        sources_json TEXT NOT NULL,
                        opened_at REAL NOT NULL,
                        horizon_s REAL NOT NULL,
                        context_json TEXT,
                        observed REAL,
                        resolved_at REAL,
                        status TEXT NOT NULL,
                        prediction_error REAL
                    )"""
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_receipts_status ON outcome_receipts(status)"
                )
                columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(outcome_receipts)").fetchall()
                }
                if "repeat_count" not in columns:
                    # repeat_count has been a dataclass field since collapsing
                    # was added, incremented in memory and never persisted, so
                    # every restart forgot how many opens had folded into a
                    # receipt and any query naming the column failed outright.
                    conn.execute(
                        "ALTER TABLE outcome_receipts ADD COLUMN repeat_count "
                        "INTEGER NOT NULL DEFAULT 1"
                    )
                    logger.info("outcome_ledger: added missing repeat_count column")
                if "observation" not in columns:
                    # Backfill honestly: every historical 'expired' row is an
                    # assumed zero, not a measurement, and there are a million
                    # of them. Marking them retroactively is the difference
                    # between a corpus that teaches and one that misleads.
                    conn.execute(
                        "ALTER TABLE outcome_receipts ADD COLUMN observation TEXT "
                        "NOT NULL DEFAULT 'measured'"
                    )
                    migrated = conn.execute(
                        "UPDATE outcome_receipts SET observation = 'unobserved' "
                        "WHERE status = 'expired'"
                    ).rowcount
                    logger.info(
                        "📒 [OutcomeLedger] marked %d expired receipts as unobserved "
                        "(assumed zeros, not measurements)", int(migrated or 0),
                    )
                conn.commit()
        except (sqlite3.Error, OSError) as e:
            record_degradation("outcome_ledger", e)

    def _persist(self, r: OutcomeReceipt) -> None:
        try:
            with connecting(self._connect()) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO outcome_receipts
                       (receipt_id, action, category, expected, sources_json, opened_at,
                        horizon_s, context_json, observed, resolved_at, status,
                        prediction_error, observation, repeat_count)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        r.receipt_id, r.action, r.category, r.expected,
                        json.dumps([s.as_dict() for s in r.sources]),
                        r.opened_at, r.horizon_s, json.dumps(r.context or {}),
                        r.observed, r.resolved_at, r.status, r.prediction_error,
                        r.observation, r.repeat_count,
                    ),
                )
                conn.commit()
        except (sqlite3.Error, OSError) as e:
            record_degradation("outcome_ledger", e)

    def _load_pending(self) -> None:
        try:
            now = time.time()
            with connecting(self._connect()) as conn:
                expired = conn.execute(
                    "UPDATE outcome_receipts "
                    "SET observed = 0.0, resolved_at = ?, status = 'expired', "
                    "prediction_error = (0.0 - expected), observation = 'unobserved' "
                    "WHERE status = 'pending' AND (? - opened_at) >= horizon_s",
                    (now, now),
                ).rowcount
                if expired:
                    conn.commit()
                self._startup_expired_count = int(expired or 0)
                self._pending_db_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM outcome_receipts WHERE status = 'pending'"
                    ).fetchone()[0]
                    or 0
                )
                limit = max(1, int(self.MAX_PENDING_LOAD))
                rows = conn.execute(
                    "SELECT receipt_id, action, category, expected, sources_json, opened_at, "
                    "horizon_s, context_json, repeat_count FROM outcome_receipts "
                    "WHERE status = 'pending' ORDER BY opened_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            self._pending_load_truncated = self._pending_db_count > len(rows)
            for rid, action, cat, exp, sj, opened, horizon, cj, repeat_count in rows:
                sources = [CreditSource(**s) for s in json.loads(sj or "[]")]
                receipt = OutcomeReceipt(
                    receipt_id=rid, action=action, category=cat, expected=exp,
                    sources=sources, opened_at=opened, horizon_s=horizon,
                    context=json.loads(cj or "{}"), repeat_count=max(1, int(repeat_count or 1)),
                )
                self._pending[rid] = receipt
                self._open_index[
                    (cat, action, self._collapse_context_key(receipt.context))
                ] = (rid, opened)
            if self._startup_expired_count:
                logger.info(
                    "📒 [OutcomeLedger] expired %d stale pending receipts during startup compaction",
                    self._startup_expired_count,
                )
            if self._pending:
                suffix = (
                    f" (loaded newest {len(self._pending)} of {self._pending_db_count})"
                    if self._pending_load_truncated
                    else ""
                )
                logger.info("📒 [OutcomeLedger] recovered %d pending receipts%s", len(self._pending), suffix)
        except (sqlite3.Error, OSError, ValueError) as e:
            record_degradation("outcome_ledger", e)

    def _hydrate_receipt_row(self, row: tuple[Any, ...]) -> OutcomeReceipt:
        rid, action, cat, exp, sj, opened, horizon, cj, repeat_count = row
        sources = [CreditSource(**s) for s in json.loads(sj or "[]")]
        return OutcomeReceipt(
            receipt_id=rid,
            action=action,
            category=cat,
            expected=exp,
            sources=sources,
            opened_at=opened,
            horizon_s=horizon,
            context=json.loads(cj or "{}"),
            repeat_count=max(1, int(repeat_count or 1)),
        )

    def _fetch_pending_receipt(self, receipt_id: str) -> OutcomeReceipt | None:
        try:
            with connecting(self._connect()) as conn:
                row = conn.execute(
                    "SELECT receipt_id, action, category, expected, sources_json, opened_at, "
                    "horizon_s, context_json, repeat_count FROM outcome_receipts "
                    "WHERE status = 'pending' AND receipt_id = ?",
                    (receipt_id,),
                ).fetchone()
            if row is None:
                return None
            return self._hydrate_receipt_row(row)
        except (sqlite3.Error, OSError, ValueError) as e:
            record_degradation("outcome_ledger", e, severity="debug")
            return None

    # ── lifecycle ────────────────────────────────────────────────────────

    def open(
        self,
        action: str,
        expected: float,
        *,
        sources: list[CreditSource] | None = None,
        category: str = "action",
        horizon_s: float | None = None,
        context: dict[str, Any] | None = None,
        now: float | None = None,
        collapse_window_s: float | None = None,
    ) -> str:
        """Commit an action with its *expected* outcome; returns a receipt_id to resolve later.

        An identical action re-opened while an identical receipt is still
        pending folds into that receipt rather than creating another row. A
        stuck reflex is one fact about the world, not a million; on this
        machine one intrusion reflex had opened 990,653 receipts, which was 96%
        of the entire ledger and would have drowned every real signal in it.
        The returned id is still a valid, resolvable receipt — callers do not
        have to know this happened.
        """
        now = time.time() if now is None else now
        window = self._collapse_window if collapse_window_s is None else float(collapse_window_s)
        if window > 0:
            collapsed = self._collapse_open(action, category, context or {}, now, window)
            if collapsed is not None:
                return collapsed
        receipt = OutcomeReceipt(
            receipt_id=f"rcpt-{uuid.uuid4().hex[:12]}",
            action=action,
            category=category,
            expected=_clamp(float(expected)),
            sources=list(sources or []),
            opened_at=now,
            horizon_s=float(horizon_s if horizon_s is not None else self._default_horizon),
            context=dict(context or {}),
        )
        with self._lock:
            self._pending[receipt.receipt_id] = receipt
            self._open_index[
                (category, action, self._collapse_context_key(receipt.context))
            ] = (receipt.receipt_id, now)
            self._persist(receipt)
            self._pending_db_count += 1
        return receipt.receipt_id

    def _collapse_open(
        self,
        action: str,
        category: str,
        context: dict[str, Any],
        now: float,
        window: float,
    ) -> str | None:
        """Fold a repeat into its still-pending original, if there is one."""
        key = (category, action, self._collapse_context_key(context))
        with self._lock:
            seen = self._open_index.get(key)
            if seen is None:
                return None
            receipt_id, opened_at = seen
            if now - opened_at > window:
                self._open_index.pop(key, None)
                return None
            receipt = self._pending.get(receipt_id)
            if receipt is None:
                # Already resolved or swept: the next call opens a fresh one.
                self._open_index.pop(key, None)
                return None
            receipt.repeat_count += 1
            self._collapsed_opens += 1
            if receipt.repeat_count % 100 == 0:
                self._persist(receipt)
            return receipt_id

    @staticmethod
    def _collapse_context_key(context: dict[str, Any]) -> str:
        """Stable identity for deduplication without retaining another raw copy."""
        import hashlib

        try:
            canonical = json.dumps(context or {}, sort_keys=True, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            canonical = repr(context)
        return hashlib.blake2s(canonical.encode("utf-8"), digest_size=8).hexdigest()

    def resolve(
        self,
        receipt_id: str,
        observed: float,
        *,
        note: str = "",
        now: float | None = None,
    ) -> OutcomeReceipt | None:
        """Close a receipt with the observed outcome; assign credit to its sources.

        Returns the resolved receipt, or None if the id is unknown/already closed.
        """
        now = time.time() if now is None else now
        with self._lock:
            receipt = self._pending.pop(receipt_id, None)
            if receipt is None:
                receipt = self._fetch_pending_receipt(receipt_id)
                if receipt is None:
                    return None
            receipt.observed = _clamp(float(observed))
            receipt.resolved_at = now
            receipt.prediction_error = receipt.observed - receipt.expected
            receipt.status = "resolved"
            receipt.observation = "measured"
            if note:
                receipt.context.setdefault("notes", []).append(note)
            self._persist(receipt)
            self._pending_db_count = max(0, self._pending_db_count - 1)
            self._calib_err_sum += abs(receipt.prediction_error)
            self._calib_n += 1
        # Credit distribution happens outside the lock (it calls into other subsystems).
        self._distribute_credit(receipt)
        self._notify_resolution(receipt)
        return receipt

    def add_resolution_observer(
        self, observer: Callable[[OutcomeReceipt], None]
    ) -> None:
        """Call ``observer`` whenever a receipt closes with a MEASURED outcome.

        This is the seam that lets learners be updated by evidence arriving
        rather than by a timer. A model refreshed on a schedule is stale for
        the whole interval and does redundant work when nothing happened; one
        refreshed on resolution is exactly as current as the evidence.

        Only ``resolve`` notifies. ``sweep`` deliberately does not: an expired
        receipt's zero is an accountability convention, and waking a learner to
        tell it that nobody looked would teach it that the action failed.
        """
        with self._lock:
            if observer not in self._resolution_observers:
                self._resolution_observers.append(observer)

    def _notify_resolution(self, receipt: OutcomeReceipt) -> None:
        with self._lock:
            observers = list(self._resolution_observers)
        for observer in observers:
            try:
                observer(receipt)
            except (RuntimeError, TypeError, ValueError, AttributeError, KeyError) as exc:
                record_degradation(
                    "outcome_ledger",
                    exc,
                    severity="warning",
                    action="resolution observer failed; the receipt is still "
                    "resolved and credit was still assigned",
                )

    def sweep(self, *, now: float | None = None) -> list[OutcomeReceipt]:
        """Expire pending receipts past their horizon.

        The accountability rule stands: an action whose outcome nobody ever
        checked is treated as an unmet expectation (observed=0), so the sources
        that promised it lose standing rather than the receipt lingering forever.

        What does *not* stand is treating that assumed zero as a measurement.
        The receipt records ``observation="unobserved"``, and the ledger's own
        calibration statistic ignores it — an expectation cannot be graded
        against an outcome that was never seen, and folding assumed zeros into
        the error would make the ledger look badly calibrated in exact
        proportion to how much of the world it failed to watch. On this machine
        that was 1,000,553 of 1,031,132 receipts, so the distinction is not
        academic.
        """
        now = time.time() if now is None else now
        expired: list[OutcomeReceipt] = []
        with self._lock:
            for rid, r in list(self._pending.items()):
                if now - r.opened_at >= r.horizon_s:
                    r.observed = 0.0
                    r.resolved_at = now
                    r.prediction_error = 0.0 - r.expected
                    r.status = "expired"
                    r.observation = "unobserved"
                    self._persist(r)
                    self._pending.pop(rid, None)
                    self._pending_db_count = max(0, self._pending_db_count - 1)
                    self._unobserved_n += 1
                    expired.append(r)
        for r in expired:
            self._distribute_credit(r)
        return expired

    # ── credit ───────────────────────────────────────────────────────────

    def _distribute_credit(self, receipt: OutcomeReceipt) -> None:
        """Push credit to each contributing source, weighted by its share.

        Credit maps observed∈[0,1] to a reward∈[-1,1] (0.5 is neutral), scaled by each
        source's normalized weight. Feeds the shared CreditAssignmentSystem (per-source
        kind as the domain) and the OutcomeLearner.
        """
        observed = receipt.observed if receipt.observed is not None else 0.0
        reward = 2.0 * observed - 1.0  # [0,1] → [-1,1]

        # An outcome that carried weight (strongly good or bad) teaches the mattering model
        # that this action's topics matter — so "what matters" is learned from consequences,
        # not declared.
        try:
            from core.cognition.mattering import get_mattering_model
            get_mattering_model().note_mattered(receipt.action, weight=0.5 * abs(reward))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass

        total_w = sum(max(0.0, s.weight) for s in receipt.sources) or 1.0
        try:
            from core.consciousness.credit_assignment import get_credit_assignment_system
            credit_system = get_credit_assignment_system()
            for s in receipt.sources:
                share = max(0.0, s.weight) / total_w
                action_id = f"{s.kind}:{s.ref}"
                credit_system.assign_credit(action_id, reward * share, domain=s.kind)
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as e:
            record_degradation("outcome_ledger", e, severity="debug",
                               action="skipped credit-assignment feed")

        try:
            from core.cognition.outcome_learner import get_outcome_learner
            duration_ms = (receipt.delay or 0.0) * 1000.0
            get_outcome_learner().record_outcome(
                category=receipt.category,
                action=receipt.action,
                success=observed >= 0.5,
                confidence_before=receipt.expected,
                duration_ms=duration_ms,
                context={
                    "expected": receipt.expected,
                    "observed": observed,
                    "prediction_error": receipt.prediction_error,
                    "status": receipt.status,
                    "sources": [s.as_dict() for s in receipt.sources],
                },
            )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as e:
            record_degradation("outcome_ledger", e, severity="debug",
                               action="skipped outcome-learner feed")

    # ── readout ──────────────────────────────────────────────────────────

    def expectation_calibration(self) -> float:
        """Mean absolute prediction error across *measured* receipts.

        Deliberately excludes receipts that expired unobserved. Grading an
        expectation against an outcome nobody looked at measures the observer,
        not the expectation.
        """
        with self._lock:
            if self._calib_n == 0:
                return 0.0
            return self._calib_err_sum / self._calib_n

    def observation_rate(self) -> float:
        """Share of closed receipts whose outcome was actually seen.

        The number worth watching. A ledger that closes thousands of receipts
        and measures a handful of them is not doing credit assignment, it is
        doing bookkeeping — and this makes that visible instead of letting it
        hide inside a healthy-looking resolved count.
        """
        with self._lock:
            closed = self._calib_n + self._unobserved_n
            return (self._calib_n / closed) if closed else 0.0

    def pending(self) -> list[dict[str, Any]]:
        with self._lock:
            return [r.as_dict() for r in self._pending.values()]

    def credit_by_source(self, *, hours: int = 24, now: float | None = None) -> dict[str, float]:
        """Net reward by source ref over resolved receipts in the window (from the db)."""
        cutoff = (time.time() if now is None else now) - hours * 3600
        out: dict[str, float] = {}
        try:
            with connecting(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT sources_json, observed FROM outcome_receipts "
                    "WHERE status != 'pending' AND resolved_at > ?",
                    (cutoff,),
                ).fetchall()
            for sj, observed in rows:
                reward = 2.0 * (observed or 0.0) - 1.0
                sources = json.loads(sj or "[]")
                total_w = sum(max(0.0, s.get("weight", 1.0)) for s in sources) or 1.0
                for s in sources:
                    share = max(0.0, s.get("weight", 1.0)) / total_w
                    out[s["ref"]] = out.get(s["ref"], 0.0) + reward * share
        except (sqlite3.Error, OSError, ValueError) as e:
            record_degradation("outcome_ledger", e, severity="debug")
        return {k: round(v, 4) for k, v in out.items()}

    def measured_action_stats(
        self, *, limit: int = 20000, by_state: bool = False
    ) -> dict[str, dict[str, float]]:
        """Per-action outcome statistics over MEASURED receipts only.

        The evidence base for any learned action value. ``observation`` must be
        ``'measured'``: an expired receipt's 0.0 is an accountability
        convention, not an observation of the world, and folding those into a
        mean would teach that every unwatched action failed.

        ``repeat_count`` is honoured as a weight so one stuck reflex counts as
        one fact with a weight rather than as thousands of independent
        successes — otherwise a loop would dominate the statistics of every
        other action.

        Returns ``{action: {"n": weight, "mean": .., "m2": ..}}`` where ``m2``
        is the weighted sum of squared deviations, so a caller can compute
        within-group variance without a second pass over the table.

        ``by_state=True`` keys on ``"<state>|<action>"`` instead, using the
        ``state`` field a caller stored in the receipt context. That is the
        difference between learning "opening Notes tends to succeed" and
        "opening Notes succeeds in this situation and not that one" — V(a)
        versus Q(s,a). Contextual buckets are necessarily thinner, which is
        exactly why the consumer backs off to the marginal estimate rather
        than trusting a bucket of one.
        """
        out: dict[str, dict[str, float]] = {}
        columns = "action, observed, repeat_count" + (", context_json" if by_state else "")
        try:
            with connecting(self._connect()) as conn:
                rows = conn.execute(
                    f"SELECT {columns} FROM outcome_receipts "  # noqa: S608 - fixed literals
                    "WHERE observation = 'measured' AND observed IS NOT NULL "
                    "ORDER BY resolved_at DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
        except (sqlite3.Error, OSError, ValueError) as e:
            # Not debug. An empty return here is indistinguishable from "no
            # evidence exists", so every learned action value silently becomes
            # a prior — which is exactly how this method spent its first hours
            # returning nothing at all against a schema with no repeat_count
            # column.
            record_degradation(
                "outcome_ledger",
                e,
                severity="warning",
                action="action-value statistics unavailable; learned values "
                "degrade to the global prior",
            )
            return out

        for row in rows:
            action, observed, repeat = row[0], row[1], row[2]
            if observed is None:
                continue
            key = str(action)
            if by_state:
                state = ""
                try:
                    ctx = json.loads(row[3] or "{}")
                    state = str(ctx.get("state") or "")
                except (ValueError, TypeError):
                    state = ""
                if not state:
                    continue  # no state recorded: it belongs in the marginal table
                key = f"{state}|{key}"
            weight = max(1.0, float(repeat or 1))
            value = float(observed)
            bucket = out.setdefault(key, {"n": 0.0, "mean": 0.0, "m2": 0.0})
            # Weighted Welford: stable in one pass, and the variance is needed
            # by the shrinkage estimator that consumes this.
            total = bucket["n"] + weight
            delta = value - bucket["mean"]
            bucket["mean"] += delta * (weight / total)
            bucket["m2"] += weight * delta * (value - bucket["mean"])
            bucket["n"] = total
        return out

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "pending": len(self._pending),
                "pending_db_count": self._pending_db_count,
                "startup_expired_count": self._startup_expired_count,
                "pending_load_truncated": self._pending_load_truncated,
                "measured_count": self._calib_n,
                "unobserved_count": self._unobserved_n,
                "observation_rate": round(self.observation_rate(), 4),
                "collapsed_opens": self._collapsed_opens,
                "expectation_calibration": round(self.expectation_calibration(), 4),
                "db_path": self._db_path,
            }


_ledger: OutcomeLedger | None = None
_ledger_lock = threading.Lock()


def get_outcome_ledger() -> OutcomeLedger:
    global _ledger
    if _ledger is None:
        with _ledger_lock:
            if _ledger is None:
                _ledger = OutcomeLedger()
    return _ledger
