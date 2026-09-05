"""L0 — the experience spine: an honest record of what Aura decided and what came of it.

Everything above this file is astrology if this file lies. Three specific lies
were already in the live record when this was written, and each one is fatal to
a learner in a different way:

**Unobserved labeled as failure.** The outcome ledger sweeps a receipt past its
horizon by writing ``observed = 0.0`` — defensible for accountability, ruinous
as a training label. 1,000,553 of 1,031,132 live receipts were swept that way.
A model fitted to that learns that Aura fails at everything. Here an outcome is
three-valued and ``UNOBSERVED`` is a first-class terminal state: it is never
coerced to failure, and it is excluded from training rather than counted as a
zero. Not knowing is a fact about the *observer*, not about the action.

**Repetition drowning signal.** 990,653 of those receipts were one reflex
firing over and over. Ninety-six percent of the corpus was a single fact
repeated. Bursts of identical episodes are rate-limited into one row with a
``repeat_count``, so a stuck reflex costs one row and one honest weight instead
of a million rows and a wrecked prior — while ordinary traffic, where two
similar-looking decisions may still turn out differently, is left alone.

**Test data in the live corpus.** Thirty thousand synthetic benchmark rows sat
in the live ledger, indistinguishable from lived experience. Provenance here is
structural, not conventional: a store rooted under the live data directory
physically refuses any write that is not ``Provenance.LIVE``, so a test cannot
contaminate the corpus by forgetting to set a flag.

Writes are queued and flushed by a background thread. The live path calls
``record()`` on the executive's hot loop; an on-loop sqlite fsync once froze
Aura's event loop for twenty minutes, and this organ is never worth that.
Overflow drops the *oldest* pending episodes and says so — losing recent
experience to preserve stale experience would be exactly backwards.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import LockRank, checked_lock
from core.runtime.sqlite_support import connecting
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.Ontogeny.Experience")

SPINE_SCHEMA = "aura.ontogeny.experience.v1"

#: Pending episodes held in memory before a flush. Roughly a minute of the
#: busiest observed decision rate; past this the oldest are dropped.
_QUEUE_CAPACITY = 4096

#: Flush cadence. Long enough to batch, short enough that a crash loses
#: seconds of experience rather than minutes.
_FLUSH_INTERVAL_S = 2.0

#: Window over which identical episodes are counted as a burst.
_DEDUP_WINDOW_S = 60.0

#: Identical episodes tolerated inside the window before the rest collapse.
#:
#: This is a rate limit, not deduplication, and the distinction matters. Two
#: genuinely different intents can present identical features and get the same
#: verdict, and they will be graded separately — merging them would throw away
#: the disagreement between their outcomes, which is exactly the signal worth
#: having. What must be clamped is the runaway case: one reflex firing
#: thousands of times with nothing changing, which on this machine produced
#: 990,653 identical receipts. Below the threshold every episode is its own
#: row; above it, the loop costs one row and an honest count.
_DEDUP_BURST_THRESHOLD = 5

#: Feature quantisation for the dedup key. Two decimals: a reflex firing with
#: bit-identical context collapses, a genuinely different situation does not.
_DEDUP_PRECISION = 2

#: Rows retained in the live store. Older resolved episodes are compacted into
#: the running sufficient statistics the heads actually train from, then
#: dropped. The corpus is bounded; the learning is not.
_RETENTION_ROWS = 500_000

#: How long a corpus-count scan is reused. stats() runs three unindexed
#: aggregates over the whole episodes table and the health report calls it
#: once per control point; without this, one report scanned the corpus N
#: times and the flush lock served it. Writes invalidate, so the only
#: staleness this can show is a count that lags by under a flush interval.
_STATS_TTL_S = 10.0


class OutcomeKind(StrEnum):
    """What actually happened — with 'we do not know' as a real answer.

    The distinction between FAILURE and UNOBSERVED is the whole point. A
    decision whose consequence was never observed carries no evidence about
    the decision, and a corpus that cannot say so teaches pessimism.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    UNOBSERVED = "unobserved"

    @property
    def is_evidence(self) -> bool:
        """True when this outcome may be used as a training label."""
        return self is not OutcomeKind.UNOBSERVED


class Provenance(StrEnum):
    """Where an episode came from. Enforced structurally, not by convention."""

    #: Lived experience from the running instance. The only kind a live store accepts.
    LIVE = "live"
    #: Produced by a test. Physically refused by the live store.
    TEST = "test"
    #: Synthetic data for benchmarking a head. Never mixed with lived experience.
    BENCHMARK = "benchmark"
    #: Replayed from an archived corpus for offline training.
    REPLAY = "replay"


@dataclass(frozen=True)
class Outcome:
    """The resolved consequence of an episode."""

    kind: OutcomeKind
    #: Scalar utility in [0, 1] when the resolver could measure one. Always
    #: ``None`` for UNOBSERVED — an unmeasured outcome has no magnitude either.
    utility: float | None = None
    resolved_at: float = field(default_factory=time.time)
    #: Who resolved it and how, e.g. ``"executive.intent_complete"``. The
    #: resolver's identity is part of the evidence: a label is only as good as
    #: the thing that produced it.
    resolver: str = "unknown"
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind is OutcomeKind.UNOBSERVED and self.utility is not None:
            raise ValueError("an unobserved outcome cannot carry a utility")
        if self.utility is not None and not 0.0 <= float(self.utility) <= 1.0:
            raise ValueError(f"utility {self.utility!r} outside [0, 1]")

    @classmethod
    def unobserved(cls, resolver: str, **detail: Any) -> Outcome:
        return cls(kind=OutcomeKind.UNOBSERVED, utility=None, resolver=resolver, detail=detail)

    @classmethod
    def from_utility(cls, utility: float, resolver: str, *, threshold: float = 0.5, **detail: Any) -> Outcome:
        kind = OutcomeKind.SUCCESS if float(utility) >= threshold else OutcomeKind.FAILURE
        return cls(kind=kind, utility=float(utility), resolver=resolver, detail=detail)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "utility": self.utility,
            "resolved_at": self.resolved_at,
            "resolver": self.resolver,
            "detail": dict(self.detail),
        }


@dataclass
class Episode:
    """One decision, its context, and — eventually — what it led to.

    An episode is opened at decision time with everything that was true *then*
    and nothing that became true after. Hindsight leaking into the feature
    vector is the classic way a control learner scores brilliantly offline and
    does nothing live, so features are captured at open and are immutable.
    """

    control_point: str
    features: Mapping[str, float]
    decision: str
    options: Sequence[str] = ()
    #: Which controller actually chose, and its version:
    #: ``"incumbent:executive_rules"`` or ``"ontogeny:executive.admission@7"``.
    decider: str = "incumbent"
    #: True when a reservation forced this episode to the other decider so the
    #: counterfactual stays observed. See L1.
    exploration: bool = False
    #: The ontogeny head's predicted distribution at decision time, recorded
    #: whether or not it had authority. This is what makes shadow evaluation
    #: possible without re-running anything.
    shadow: Mapping[str, float] | None = None
    shadow_version: int | None = None
    #: How much rides on this one, 0..1. High-stakes episodes are never
    #: reserved for exploration and are weighted harder in evaluation.
    stakes: float = 0.5
    horizon_s: float = 900.0
    provenance: Provenance = Provenance.LIVE
    feature_schema: str = ""
    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    decided_at: float = field(default_factory=time.time)
    outcome: Outcome | None = None
    repeat_count: int = 1
    #: Free-form context for forensics. Never used as a feature — anything a
    #: head is allowed to learn from must be a declared, versioned feature.
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.control_point:
            raise ValueError("an episode must name its control point")
        if not self.decision:
            raise ValueError("an episode must name the decision taken")
        self.features = {str(k): float(v) for k, v in dict(self.features).items()}
        self.options = tuple(self.options) or (self.decision,)
        if self.decision not in self.options:
            self.options = (*self.options, self.decision)
        self.stakes = max(0.0, min(1.0, float(self.stakes)))
        if not self.feature_schema:
            self.feature_schema = schema_id(self.features.keys())

    @property
    def dedup_key(self) -> str:
        """Identity for collapsing a stuck loop into one honest row."""
        quantised = ",".join(
            f"{k}={round(float(v), _DEDUP_PRECISION)}"
            for k, v in sorted(self.features.items())
        )
        payload = f"{self.control_point}|{self.decision}|{self.decider}|{quantised}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "control_point": self.control_point,
            "decided_at": self.decided_at,
            "features": dict(self.features),
            "feature_schema": self.feature_schema,
            "decision": self.decision,
            "options": list(self.options),
            "decider": self.decider,
            "exploration": self.exploration,
            "shadow": dict(self.shadow) if self.shadow else None,
            "shadow_version": self.shadow_version,
            "stakes": self.stakes,
            "horizon_s": self.horizon_s,
            "provenance": str(self.provenance),
            "repeat_count": self.repeat_count,
            "outcome": self.outcome.as_dict() if self.outcome else None,
        }


def schema_id(names: Iterable[str]) -> str:
    """Stable id for a feature set.

    Features are keyed by name, and names come from subsystems that get
    renamed. A schema id lets a row from before a rename be *invalidated*
    rather than silently misinterpreted as the new meaning of the same slot.
    """
    joined = "|".join(sorted(str(n) for n in names))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


class ExperienceSpine:
    """The corpus. Append-fast, resolve-later, bounded, and provenance-gated."""

    def __init__(self, db_path: str | Path | None = None, *, autoflush: bool = True) -> None:
        self._db_path = Path(db_path) if db_path else _default_db_path()
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope(
            "ontogeny_experience",
            domain="state_mutation",
            receipt_prefix="ontogeny-experience",
        ):
            get_file_write_gateway().ensure_directory(
                self._db_path.parent,
                source="ontogeny_experience",
            )
        self._store_kind = _classify_store(self._db_path)
        self._lock = checked_lock("ontogeny.spine", rank=LockRank.LEAF, reentrant=True)
        self._queue: deque[Episode] = deque()
        self._pending_resolutions: deque[tuple[str, Outcome]] = deque()
        self._repeat_increments: deque[str] = deque()
        self._resolve_callbacks: list[Callable[[str, Outcome], None]] = []
        self._dedup: dict[str, tuple[str, float]] = {}
        self._burst: dict[str, int] = {}
        self._dropped = 0
        self._written = 0
        self._collapsed = 0
        self._refused = 0
        self._stopped = threading.Event()
        self._flusher: threading.Thread | None = None
        # stats() is three unindexed aggregates over the whole episodes table,
        # and ontogeny_report() calls it once per control point — so a single
        # health report used to scan the corpus N times. Under demo load that
        # was the leaf of a 103.8s event-loop stall (2026-07-29). Counts move
        # slowly and nothing reads them for control, so serve them from a
        # short TTL and let a write invalidate.
        self._stats_cache: dict[str | None, tuple[float, dict[str, Any]]] = {}
        self._observation_stats_cache: dict[
            tuple[str, int], tuple[float, dict[str, Any]]
        ] = {}
        self._init_schema()
        if autoflush:
            self._start_flusher()

    # ── properties ───────────────────────────────────────────────────────

    @property
    def store_kind(self) -> str:
        """``"live"`` for the instance's own corpus, ``"sandbox"`` otherwise."""
        return self._store_kind

    @property
    def db_path(self) -> Path:
        return self._db_path

    # ── schema ───────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        try:
            with connecting(self._connect()) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS episodes (
                        episode_id      TEXT PRIMARY KEY,
                        control_point   TEXT NOT NULL,
                        decided_at      REAL NOT NULL,
                        features_json   TEXT NOT NULL,
                        feature_schema  TEXT NOT NULL,
                        decision        TEXT NOT NULL,
                        options_json    TEXT NOT NULL,
                        decider         TEXT NOT NULL,
                        exploration     INTEGER NOT NULL DEFAULT 0,
                        shadow_json     TEXT,
                        shadow_version  INTEGER,
                        stakes          REAL NOT NULL DEFAULT 0.5,
                        horizon_s       REAL NOT NULL DEFAULT 900.0,
                        provenance      TEXT NOT NULL,
                        dedup_key       TEXT NOT NULL,
                        repeat_count    INTEGER NOT NULL DEFAULT 1,
                        context_json    TEXT,
                        outcome_kind    TEXT,
                        outcome_utility REAL,
                        resolved_at     REAL,
                        resolver        TEXT,
                        outcome_detail  TEXT
                    )
                    """
                )
                for stmt in (
                    "CREATE INDEX IF NOT EXISTS idx_ep_cp_time ON episodes(control_point, decided_at)",
                    "CREATE INDEX IF NOT EXISTS idx_ep_open ON episodes(outcome_kind, decided_at)",
                    "CREATE INDEX IF NOT EXISTS idx_ep_dedup ON episodes(dedup_key, decided_at)",
                ):
                    conn.execute(stmt)
        except sqlite3.Error as exc:
            record_degradation(
                "ontogeny_experience", exc,
                action="experience spine schema unavailable; episodes will not persist",
            )

    # ── the hot path ─────────────────────────────────────────────────────

    def record(self, episode: Episode) -> str | None:
        """Queue an episode. Returns its id, or ``None`` if refused or dropped.

        Called from the executive's decision path, so it does no I/O: the row
        is queued and a background thread writes it.
        """
        if not self._accepts(episode.provenance):
            self._refused += 1
            return None
        with self._lock:
            collapsed_into = self._collapse_target(episode)
            if collapsed_into is not None:
                self._collapsed += 1
                return collapsed_into
            if len(self._queue) >= _QUEUE_CAPACITY:
                self._queue.popleft()
                self._dropped += 1
                if self._dropped in (1, 100, 1000) or self._dropped % 10_000 == 0:
                    logger.warning(
                        "ontogeny: experience queue saturated, dropped %d oldest episodes",
                        self._dropped,
                    )
            self._queue.append(episode)
            self._dedup.setdefault(episode.dedup_key, (episode.episode_id, episode.decided_at))
            return episode.episode_id

    def _accepts(self, provenance: Provenance) -> bool:
        """A live store takes lived experience and nothing else.

        This is the structural half of the guarantee: a test that forgets to
        redirect its store does not quietly poison the corpus, it gets nothing
        written and a refusal counter it can assert on.
        """
        if self._store_kind == "live":
            return provenance is Provenance.LIVE
        return True

    def _collapse_target(self, episode: Episode) -> str | None:
        """Clamp a runaway loop, without merging decisions that get graded apart.

        The first few identical episodes in a window each get their own row —
        two different intents can look identical to the organ and still have
        different outcomes, and that disagreement is signal. Only once the same
        key keeps arriving does it start folding.
        """
        seen = self._dedup.get(episode.dedup_key)
        if seen is None:
            self._burst[episode.dedup_key] = 1
            return None
        original_id, first_at = seen
        if episode.decided_at - first_at > _DEDUP_WINDOW_S:
            self._burst[episode.dedup_key] = 1
            return None
        count = self._burst.get(episode.dedup_key, 0) + 1
        self._burst[episode.dedup_key] = count
        if count <= _DEDUP_BURST_THRESHOLD:
            return None
        for queued in reversed(self._queue):
            if queued.episode_id == original_id:
                queued.repeat_count += 1
                return original_id
        # Already flushed: the increment rides along with the next batch.
        self._repeat_increments.append(original_id)
        return original_id

    def on_resolve(self, callback: Callable[[str, Outcome], None]) -> None:
        """Subscribe to outcomes as they land.

        Every resolution in the system passes through :meth:`resolve`, which
        makes this the one place a live tally can be kept honest without
        polling the database from a decision path.
        """
        self._resolve_callbacks.append(callback)

    def resolve(self, episode_id: str, outcome: Outcome) -> None:
        """Attach an outcome. Queued like a record — resolution is never urgent."""
        with self._lock:
            queued_hit = False
            for queued in self._queue:
                if queued.episode_id == episode_id:
                    queued.outcome = outcome
                    queued_hit = True
                    break
            if not queued_hit:
                self._pending_resolutions.append((episode_id, outcome))
        for callback in tuple(self._resolve_callbacks):
            try:
                callback(episode_id, outcome)
            except (RuntimeError, ValueError, TypeError, AttributeError, KeyError) as exc:
                record_degradation(
                    "ontogeny_experience", exc, severity="warning",
                    action="resolve subscriber raised; the outcome was still recorded",
                )

    # ── flushing ─────────────────────────────────────────────────────────

    def _start_flusher(self) -> None:
        if self._flusher is not None:
            return
        self._flusher = threading.Thread(
            target=self._flush_loop, name="ontogeny-experience-flush", daemon=True
        )
        self._flusher.start()

    def _flush_loop(self) -> None:
        while not self._stopped.wait(_FLUSH_INTERVAL_S):
            try:
                self.flush()
            except (sqlite3.Error, OSError) as exc:
                record_degradation(
                    "ontogeny_experience", exc, action="experience flush failed; retrying next cycle"
                )

    def flush(self) -> int:
        """Write queued episodes and resolutions. Returns rows written."""
        with self._lock:
            batch = list(self._queue)
            resolutions = list(self._pending_resolutions)
            repeats = list(self._repeat_increments)
            self._queue.clear()
            self._pending_resolutions.clear()
            self._repeat_increments.clear()
            self._prune_dedup()
        if not batch and not resolutions and not repeats:
            return 0
        try:
            with connecting(self._connect()) as conn:
                if batch:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO episodes (
                            episode_id, control_point, decided_at, features_json,
                            feature_schema, decision, options_json, decider, exploration,
                            shadow_json, shadow_version, stakes, horizon_s, provenance,
                            dedup_key, repeat_count, context_json,
                            outcome_kind, outcome_utility, resolved_at, resolver, outcome_detail
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        [self._row(ep) for ep in batch],
                    )
                for episode_id, outcome in resolutions:
                    conn.execute(
                        """
                        UPDATE episodes SET outcome_kind=?, outcome_utility=?,
                               resolved_at=?, resolver=?, outcome_detail=?
                        WHERE episode_id=? AND outcome_kind IS NULL
                        """,
                        (
                            str(outcome.kind), outcome.utility, outcome.resolved_at,
                            outcome.resolver, json.dumps(dict(outcome.detail))[:4000],
                            episode_id,
                        ),
                    )
                for episode_id in repeats:
                    conn.execute(
                        "UPDATE episodes SET repeat_count = repeat_count + 1 WHERE episode_id=?",
                        (episode_id,),
                    )
            self._written += len(batch)
            self._stats_cache.clear()
            self._observation_stats_cache.clear()
            return len(batch)
        except sqlite3.Error as exc:
            record_degradation(
                "ontogeny_experience", exc, action="experience batch lost; corpus continues"
            )
            return 0

    @staticmethod
    def _row(ep: Episode) -> tuple:
        outcome = ep.outcome
        return (
            ep.episode_id, ep.control_point, ep.decided_at,
            json.dumps(dict(ep.features)), ep.feature_schema, ep.decision,
            json.dumps(list(ep.options)), ep.decider, int(ep.exploration),
            json.dumps(dict(ep.shadow)) if ep.shadow else None, ep.shadow_version,
            ep.stakes, ep.horizon_s, str(ep.provenance), ep.dedup_key, ep.repeat_count,
            json.dumps(dict(ep.context))[:4000] if ep.context else None,
            str(outcome.kind) if outcome else None,
            outcome.utility if outcome else None,
            outcome.resolved_at if outcome else None,
            outcome.resolver if outcome else None,
            json.dumps(dict(outcome.detail))[:4000] if outcome else None,
        )

    def _prune_dedup(self) -> None:
        cutoff = time.time() - _DEDUP_WINDOW_S
        stale = [k for k, (_, seen_at) in self._dedup.items() if seen_at < cutoff]
        for key in stale:
            self._dedup.pop(key, None)
            self._burst.pop(key, None)

    # ── reading ──────────────────────────────────────────────────────────

    def episodes(
        self,
        control_point: str | None = None,
        *,
        since: float | None = None,
        evidence_only: bool = False,
        limit: int = 10_000,
        feature_schema: str | None = None,
    ) -> list[Episode]:
        """Read episodes back.

        ``evidence_only`` returns only episodes whose outcome is real evidence
        — never the unobserved ones. Training paths must use it; forensics
        should not.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if control_point:
            clauses.append("control_point = ?")
            params.append(control_point)
        if since is not None:
            clauses.append("decided_at >= ?")
            params.append(since)
        if feature_schema:
            clauses.append("feature_schema = ?")
            params.append(feature_schema)
        if evidence_only:
            clauses.append("outcome_kind IN (?, ?)")
            params.extend([str(OutcomeKind.SUCCESS), str(OutcomeKind.FAILURE)])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        try:
            with connecting(self._connect()) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    f"SELECT * FROM episodes {where} ORDER BY decided_at DESC LIMIT ?",
                    (*params, int(limit)),
                ).fetchall()
        except sqlite3.Error as exc:
            record_degradation("ontogeny_experience", exc, severity="warning",
                               action="episode read failed; returning empty")
            return []
        return [_episode_from_row(r) for r in rows]

    def open_episodes(self, *, older_than_horizon: bool = True, limit: int = 2000) -> list[Episode]:
        """Episodes still awaiting an outcome, oldest first.

        With ``older_than_horizon`` these are the ones whose horizon has
        elapsed — the sweeper's input. Note what the sweeper does *not* do:
        it does not write them off as failures.
        """
        now = time.time()
        try:
            with connecting(self._connect()) as conn:
                conn.row_factory = sqlite3.Row
                if older_than_horizon:
                    rows = conn.execute(
                        "SELECT * FROM episodes WHERE outcome_kind IS NULL "
                        "AND (decided_at + horizon_s) <= ? ORDER BY decided_at ASC LIMIT ?",
                        (now, int(limit)),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM episodes WHERE outcome_kind IS NULL "
                        "ORDER BY decided_at ASC LIMIT ?",
                        (int(limit),),
                    ).fetchall()
        except sqlite3.Error as exc:
            record_degradation("ontogeny_experience", exc, severity="warning",
                               action="open-episode read failed")
            return []
        return [_episode_from_row(r) for r in rows]

    def stats(self, control_point: str | None = None) -> dict[str, Any]:
        """Counts by outcome kind — the honest denominator for every claim above.

        Served from a short TTL cache: the aggregates are full-table scans and
        the health report asks for them once per control point. The live
        counters (queued/dropped/written) are always read fresh, so the
        cache never makes the queue look emptier than it is.
        """
        cached = self._stats_cache.get(control_point)
        if cached is not None and (time.time() - cached[0]) < _STATS_TTL_S:
            return self._stats_with_live_counters(cached[1])
        clause, params = ("WHERE control_point = ?", [control_point]) if control_point else ("", [])
        try:
            with connecting(self._connect()) as conn:
                total, repeats = conn.execute(
                    f"SELECT COUNT(*), COALESCE(SUM(repeat_count), 0) FROM episodes {clause}", params
                ).fetchone()
                by_kind = dict(
                    conn.execute(
                        f"SELECT COALESCE(outcome_kind, 'pending'), COUNT(*) FROM episodes "
                        f"{clause} GROUP BY 1", params
                    ).fetchall()
                )
                explored = conn.execute(
                    f"SELECT COUNT(*) FROM episodes {clause}{' AND' if clause else 'WHERE'} exploration = 1",
                    params,
                ).fetchone()[0]
        except sqlite3.Error as exc:
            record_degradation("ontogeny_experience", exc, severity="warning",
                               action="experience stats unavailable")
            return {"available": False}
        evidence = int(by_kind.get(str(OutcomeKind.SUCCESS), 0)) + int(by_kind.get(str(OutcomeKind.FAILURE), 0))
        scanned = {
            "available": True,
            "store_kind": self._store_kind,
            "rows": int(total or 0),
            "observations": int(repeats or 0),
            "by_outcome": {k: int(v) for k, v in by_kind.items()},
            "evidence_rows": evidence,
            "exploration_rows": int(explored or 0),
        }
        self._stats_cache[control_point] = (time.time(), scanned)
        return self._stats_with_live_counters(scanned)

    def observation_stats(
        self,
        control_point: str,
        *,
        recent_limit: int = 500,
    ) -> dict[str, Any]:
        """Observed share in a bounded, durable window for one control point.

        ResolverRegistry counters describe only this process. They restart at
        zero and therefore cannot govern persisted authority after a reboot.
        This query reads the latest closed episodes from the corpus that
        granted that authority. It uses the control-point/time index and a
        strict row bound, so health sampling never scans the full ledger.
        """

        name = str(control_point or "").strip()
        if not name:
            raise ValueError("control_point is required")
        limit = max(1, min(5_000, int(recent_limit)))
        cache_key = (name, limit)
        cached = self._observation_stats_cache.get(cache_key)
        if cached is not None and (time.time() - cached[0]) < _STATS_TTL_S:
            return dict(cached[1])
        try:
            with connecting(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT outcome_kind, decided_at, resolved_at "
                    "FROM episodes "
                    "WHERE control_point = ? AND outcome_kind IS NOT NULL "
                    "ORDER BY resolved_at DESC, decided_at DESC LIMIT ?",
                    (name, limit),
                ).fetchall()
        except sqlite3.Error as exc:
            record_degradation(
                "ontogeny_experience",
                exc,
                severity="warning",
                action="authority observation window unavailable",
            )
            return {
                "available": False,
                "control_point": name,
                "window_limit": limit,
            }

        observed_kinds = {str(OutcomeKind.SUCCESS), str(OutcomeKind.FAILURE)}
        observed = sum(1 for kind, _decided, _resolved in rows if kind in observed_kinds)
        unobserved = sum(
            1
            for kind, _decided, _resolved in rows
            if kind == str(OutcomeKind.UNOBSERVED)
        )
        closed = len(rows)
        resolved = [float(row[2]) for row in rows if row[2] is not None]
        report = {
            "available": True,
            "control_point": name,
            "window_limit": limit,
            "closed": closed,
            "observed": observed,
            "unobserved": unobserved,
            "observation_rate": observed / closed if closed else None,
            "window_started_at": min(resolved) if resolved else None,
            "window_ended_at": max(resolved) if resolved else None,
        }
        self._observation_stats_cache[cache_key] = (time.time(), report)
        return dict(report)

    def _stats_with_live_counters(self, scanned: dict[str, Any]) -> dict[str, Any]:
        """Cached aggregates plus the in-memory counters, which are always current."""
        return {
            **scanned,
            "queued": len(self._queue),
            "dropped": self._dropped,
            "collapsed": self._collapsed,
            "refused_provenance": self._refused,
            "written": self._written,
        }

    def compact(self, *, retention_rows: int = _RETENTION_ROWS) -> int:
        """Bound the corpus. Drops the oldest resolved rows past the retention line."""
        try:
            with connecting(self._connect()) as conn:
                total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
                if total <= retention_rows:
                    return 0
                excess = int(total - retention_rows)
                cur = conn.execute(
                    "DELETE FROM episodes WHERE episode_id IN ("
                    "  SELECT episode_id FROM episodes WHERE outcome_kind IS NOT NULL "
                    "  ORDER BY decided_at ASC LIMIT ?)",
                    (excess,),
                )
                self._stats_cache.clear()
                self._observation_stats_cache.clear()
                return int(cur.rowcount or 0)
        except sqlite3.Error as exc:
            record_degradation("ontogeny_experience", exc, severity="warning",
                               action="experience compaction skipped")
            return 0

    def close(self) -> None:
        self._stopped.set()
        try:
            self.flush()
        except (sqlite3.Error, OSError) as exc:
            record_degradation("ontogeny_experience", exc, severity="warning",
                               action="final experience flush failed")


def _episode_from_row(row: sqlite3.Row) -> Episode:
    outcome = None
    if row["outcome_kind"]:
        try:
            detail = json.loads(row["outcome_detail"] or "{}")
        except (ValueError, TypeError):
            detail = {}
        outcome = Outcome(
            kind=OutcomeKind(row["outcome_kind"]),
            utility=row["outcome_utility"],
            resolved_at=row["resolved_at"] or 0.0,
            resolver=row["resolver"] or "unknown",
            detail=detail,
        )
    episode = Episode(
        control_point=row["control_point"],
        features=json.loads(row["features_json"]),
        decision=row["decision"],
        options=tuple(json.loads(row["options_json"])),
        decider=row["decider"],
        exploration=bool(row["exploration"]),
        shadow=json.loads(row["shadow_json"]) if row["shadow_json"] else None,
        shadow_version=row["shadow_version"],
        stakes=row["stakes"],
        horizon_s=row["horizon_s"],
        provenance=Provenance(row["provenance"]),
        feature_schema=row["feature_schema"],
        episode_id=row["episode_id"],
        decided_at=row["decided_at"],
    )
    episode.outcome = outcome
    episode.repeat_count = int(row["repeat_count"] or 1)
    return episode


def _default_db_path() -> Path:
    override = os.environ.get("AURA_ONTOGENY_DB")
    if override:
        return Path(override).expanduser()
    try:
        from core.config import config

        return Path(config.paths.data_dir) / "ontogeny" / "experience.db"
    except (ImportError, AttributeError, RuntimeError, OSError) as exc:
        record_degradation("ontogeny_experience", exc, severity="debug",
                           action="config paths unavailable; using home fallback")
        return state_root() / "data" / "ontogeny" / "experience.db"


def _classify_store(path: Path) -> str:
    """A store under the instance's own data directory holds lived experience."""
    try:
        from core.config import config

        live_root = Path(config.paths.data_dir).resolve()
    except (ImportError, AttributeError, RuntimeError, OSError):
        live_root = (state_root() / "data").resolve()
    try:
        resolved = path.resolve()
    except OSError:
        return "sandbox"
    return "live" if resolved.is_relative_to(live_root) else "sandbox"


_spine: ExperienceSpine | None = None
_spine_lock = threading.Lock()


def get_experience_spine() -> ExperienceSpine:
    global _spine
    if _spine is None:
        with _spine_lock:
            if _spine is None:
                _spine = ExperienceSpine()
    return _spine


def reset_experience_spine_for_test(spine: ExperienceSpine | None = None) -> None:
    """Swap the process-wide spine. Tests only; the live path never calls this."""
    global _spine
    with _spine_lock:
        if _spine is not None and spine is not _spine:
            _spine.close()
        _spine = spine


__all__ = [
    "Episode",
    "ExperienceSpine",
    "Outcome",
    "OutcomeKind",
    "Provenance",
    "SPINE_SCHEMA",
    "get_experience_spine",
    "reset_experience_spine_for_test",
    "schema_id",
]
