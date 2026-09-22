"""core/resilience/fault_evidence.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Empirical fault evidence: the FMEA's static probability estimates checked
against what the runtime actually observes, accumulated across boots.

A fault catalog whose probability column never updates is a guess frozen at
authoring time. This layer closes that gap:

- every recorded fault occurrence increments an in-memory counter (O(1),
  hot-path safe — the degradation path calls this on every event)
- counters and observed runtime hours persist across restarts through the
  governed write gateway (atomic JSON envelope, flush rate-limited and
  never on the recording path)
- observed occurrence rates map back onto MIL-STD-882E probability bands,
  and a drift report flags every definition whose static probability
  disagrees with the evidence — including the recalibrated RPN, so risk
  re-prioritization is grounded in measurement instead of intuition

Sufficiency gate: a band is only asserted from >= MIN_OCCURRENCES events
observed over >= MIN_RUNTIME_S of accumulated runtime; below that the
verdict is "insufficient_evidence", never a guess.
"""
from __future__ import annotations

import atexit
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.resilience.fault_taxonomy import FaultDefinition

logger = logging.getLogger("Aura.FaultEvidence")

_SCHEMA_NAME = "aura.fault_evidence"
_SCHEMA_VERSION = 1

# Evidence sufficiency thresholds — below these, no band is asserted.
MIN_OCCURRENCES = 3
MIN_RUNTIME_S = 3600.0

# Minimum seconds between disk flushes; recording never touches disk.
FLUSH_MIN_INTERVAL_S = 60.0

# MIL-STD-882E probability bands as occurrence-rate floors (events/second).
# Ordered most-frequent first; the first floor the observed rate meets wins.
_BAND_FLOORS: tuple[tuple[str, float], ...] = (
    ("FREQUENT", 1.0 / 86_400.0),        # >= 1/day
    ("PROBABLE", 1.0 / 604_800.0),       # >= 1/week
    ("OCCASIONAL", 1.0 / 2_592_000.0),   # >= 1/month
    ("REMOTE", 1.0 / 31_536_000.0),      # >= 1/year
    ("IMPROBABLE", 0.0),                 # < 1/year
)

_BAND_TO_LEVEL = {
    "FREQUENT": 5, "PROBABLE": 4, "OCCASIONAL": 3, "REMOTE": 2, "IMPROBABLE": 1,
}


def rate_to_band(rate_per_s: float) -> str:
    """Map an observed occurrence rate onto a MIL-STD-882E probability band."""
    for band, floor in _BAND_FLOORS:
        if rate_per_s >= floor and floor > 0.0:
            return band
    return "IMPROBABLE"


@dataclass
class FaultOccurrenceEvidence:
    """Accumulated occurrence evidence for one fault ID."""
    count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FaultOccurrenceEvidence:
        return cls(
            count=max(0, int(data.get("count", 0) or 0)),
            first_seen=float(data.get("first_seen", 0.0) or 0.0),
            last_seen=float(data.get("last_seen", 0.0) or 0.0),
        )


@dataclass
class DriftFinding:
    """One definition whose static probability disagrees with the evidence."""
    fault_id: str
    name: str
    static_band: str
    implied_band: str
    level_delta: int                # implied level - static level (+ = riskier)
    occurrences: int
    observed_rate_per_day: float
    static_rpn: int
    recalibrated_rpn: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "name": self.name,
            "static_probability": self.static_band,
            "implied_probability": self.implied_band,
            "level_delta": self.level_delta,
            "occurrences": self.occurrences,
            "observed_rate_per_day": round(self.observed_rate_per_day, 4),
            "static_rpn": self.static_rpn,
            "recalibrated_rpn": self.recalibrated_rpn,
        }


class FaultEvidenceStore:
    """Cross-boot fault occurrence evidence with governed persistence.

    Thread-safe. record() is lock-guarded counter arithmetic only; all
    filesystem work happens in flush(), which callers invoke off the hot
    path (diagnostics access, shutdown, atexit).
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._evidence: dict[str, FaultOccurrenceEvidence] = {}
        self._prior_runtime_s = 0.0     # accumulated across previous boots
        self._session_started = time.monotonic()
        self._last_flush = 0.0
        self._dirty = False
        self._load()

    # ── recording (hot-path safe) ────────────────────────────────────

    def record(self, fault_id: str) -> None:
        now = time.time()
        with self._lock:
            ev = self._evidence.get(fault_id)
            if ev is None:
                ev = FaultOccurrenceEvidence(first_seen=now)
                self._evidence[fault_id] = ev
            ev.count += 1
            ev.last_seen = now
            self._dirty = True

    # ── observed-runtime accounting ──────────────────────────────────

    def observed_runtime_s(self) -> float:
        with self._lock:
            return self._prior_runtime_s + (time.monotonic() - self._session_started)

    # ── rate + band queries ──────────────────────────────────────────

    def occurrence_count(self, fault_id: str) -> int:
        with self._lock:
            ev = self._evidence.get(fault_id)
            return ev.count if ev else 0

    def implied_probability(self, fault_id: str) -> tuple[str, str]:
        """Return (band, basis) for a fault ID.

        basis is "measured" when the sufficiency gate passes, otherwise
        "insufficient_evidence" and the band is what the data would imply
        so far (advisory only).
        """
        runtime = self.observed_runtime_s()
        with self._lock:
            ev = self._evidence.get(fault_id)
            count = ev.count if ev else 0
        if runtime <= 0.0:
            return ("IMPROBABLE", "insufficient_evidence")
        band = rate_to_band(count / runtime)
        if count >= MIN_OCCURRENCES and runtime >= MIN_RUNTIME_S:
            return (band, "measured")
        return (band, "insufficient_evidence")

    def drift_report(self, definitions: list[FaultDefinition]) -> list[DriftFinding]:
        """Definitions whose static probability disagrees with measurement.

        Only measured (sufficiency-gated) disagreements are reported; the
        report is sorted by absolute recalibrated-RPN change, largest first.
        """
        findings: list[DriftFinding] = []
        runtime = self.observed_runtime_s()
        if runtime <= 0.0:
            return findings
        for defn in definitions:
            band, basis = self.implied_probability(defn.fault_id)
            if basis != "measured":
                continue
            static_band = defn.probability.name
            if band == static_band:
                continue
            implied_level = _BAND_TO_LEVEL[band]
            recalibrated = int(defn.severity) * implied_level * int(defn.detection)
            count = self.occurrence_count(defn.fault_id)
            findings.append(DriftFinding(
                fault_id=defn.fault_id,
                name=defn.name,
                static_band=static_band,
                implied_band=band,
                level_delta=implied_level - _BAND_TO_LEVEL[static_band],
                occurrences=count,
                observed_rate_per_day=count / runtime * 86_400.0,
                static_rpn=defn.rpn,
                recalibrated_rpn=recalibrated,
            ))
        findings.sort(key=lambda f: abs(f.recalibrated_rpn - f.static_rpn), reverse=True)
        return findings

    def status(self) -> dict[str, Any]:
        with self._lock:
            tracked = len(self._evidence)
            total = sum(ev.count for ev in self._evidence.values())
        return {
            "tracked_fault_ids": tracked,
            "total_occurrences": total,
            "observed_runtime_s": round(self.observed_runtime_s(), 1),
            "evidence_path": str(self.path),
            "min_occurrences": MIN_OCCURRENCES,
            "min_runtime_s": MIN_RUNTIME_S,
        }

    # ── persistence (never on the recording path) ────────────────────

    def _load(self) -> None:
        try:
            envelope = json.loads(self.path.read_text(encoding="utf-8"))
        # not a failure: what is already gone is the state this is reaching.
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            # Corrupt evidence resets to empty — losing priors is safe;
            # inventing them is not.
            logger.warning("Fault evidence unreadable, starting fresh: %s", exc)
            return
        payload = envelope.get("payload") if isinstance(envelope, dict) else None
        if not isinstance(payload, dict):
            return
        try:
            self._prior_runtime_s = max(0.0, float(payload.get("observed_runtime_s", 0.0)))
            for fault_id, data in (payload.get("evidence") or {}).items():
                if isinstance(data, dict):
                    self._evidence[str(fault_id)] = FaultOccurrenceEvidence.from_dict(data)
        except (TypeError, ValueError) as exc:
            logger.warning("Fault evidence malformed, starting fresh: %s", exc)
            self._evidence.clear()
            self._prior_runtime_s = 0.0

    def flush(self, *, force: bool = False) -> bool:
        """Persist evidence through the governed write gateway.

        Rate-limited unless forced. Returns True if a write happened.
        """
        now = time.monotonic()
        with self._lock:
            if not self._dirty and not force:
                return False
            if not force and (now - self._last_flush) < FLUSH_MIN_INTERVAL_S:
                return False
            snapshot = {
                fault_id: ev.to_dict() for fault_id, ev in self._evidence.items()
            }
            runtime = self._prior_runtime_s + (time.monotonic() - self._session_started)
            self._last_flush = now
            self._dirty = False

        payload = {
            "observed_runtime_s": round(runtime, 1),
            "evidence": snapshot,
            "flushed_at": time.time(),
        }
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            self.path.parent.mkdir(parents=True, exist_ok=True)
            with local_internal_governed_scope(
                "fault_evidence.flush",
                constraints={"purpose": "persist cross-boot fault occurrence evidence"},
            ):
                get_file_write_gateway().write_json(
                    self.path,
                    payload,
                    schema_version=_SCHEMA_VERSION,
                    schema_name=_SCHEMA_NAME,
                    source="fault_evidence",
                )
            return True
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as exc:
            logger.warning("Fault evidence flush failed: %s", exc)
            with self._lock:
                self._dirty = True  # retry on the next flush window
            return False


# ── Module singleton (lazy; enabled by the fault registry) ───────────

_store: FaultEvidenceStore | None = None
_store_lock = threading.Lock()


def default_evidence_path() -> Path:
    from core.config import DATA_DIR
    return Path(DATA_DIR) / "reliability" / "fault_evidence.json"


def get_fault_evidence_store(path: Path | str | None = None) -> FaultEvidenceStore:
    """Process-wide evidence store; first caller binds the path."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = FaultEvidenceStore(path or default_evidence_path())
                atexit.register(_store.flush, force=True)
    return _store
