"""IdentityLedger — durable record of commitments, preferences, self-model.

Audit constraint: identity continuity must survive restart, memory
compaction, and upgrades. Prompt persona is not enough.

Components:
  - CommitmentTracker: things Aura promised
  - PreferenceHistory: how preferences changed over time
  - SelfModelVersioning: snapshots of the self-model
  - ContradictionDetector: warns before saying something inconsistent
  - IdentityDriftMonitor: tracks unexplained shifts
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_json, read_json_envelope
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.IdentityLedger")


@dataclass
class Commitment:
    commitment_id: str
    text: str
    created_at: float
    fulfilled_at: float | None = None
    revoked_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreferenceChange:
    change_id: str
    key: str
    old_value: Any
    new_value: Any
    reason: str
    at: float


@dataclass
class SelfModelSnapshot:
    snapshot_id: str
    at: float
    state: dict[str, Any]


class CommitmentTracker:
    def __init__(self):
        self._commitments: dict[str, Commitment] = {}
        self._lock = threading.RLock()

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> Commitment:
        c = Commitment(
            commitment_id=f"commit-{uuid.uuid4()}",
            text=text,
            created_at=time.time(),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._commitments[c.commitment_id] = c
        return c

    def fulfill(self, commitment_id: str) -> bool:
        with self._lock:
            c = self._commitments.get(commitment_id)
            if c is None or c.fulfilled_at is not None:
                return False
            c.fulfilled_at = time.time()
            return True

    def revoke(self, commitment_id: str) -> bool:
        with self._lock:
            c = self._commitments.get(commitment_id)
            if c is None or c.revoked_at is not None:
                return False
            c.revoked_at = time.time()
            return True

    def open_commitments(self) -> list[Commitment]:
        with self._lock:
            return [
                c for c in self._commitments.values()
                if c.fulfilled_at is None and c.revoked_at is None
            ]

    def all(self) -> list[Commitment]:
        with self._lock:
            return list(self._commitments.values())


class PreferenceHistory:
    def __init__(self):
        self._history: list[PreferenceChange] = []
        self._current: dict[str, Any] = {}
        self._lock = threading.RLock()

    def set(self, key: str, value: Any, reason: str = "") -> None:
        with self._lock:
            old = self._current.get(key)
            self._history.append(
                PreferenceChange(
                    change_id=f"pref-{uuid.uuid4()}",
                    key=key,
                    old_value=old,
                    new_value=value,
                    reason=reason,
                    at=time.time(),
                )
            )
            self._current[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._current.get(key, default)

    def history_for(self, key: str) -> list[PreferenceChange]:
        with self._lock:
            return [c for c in self._history if c.key == key]

    def all_changes(self) -> list[PreferenceChange]:
        with self._lock:
            return list(self._history)


class SelfModelVersioning:
    def __init__(self):
        self._snapshots: list[SelfModelSnapshot] = []
        self._lock = threading.RLock()

    def snapshot(self, state: dict[str, Any]) -> SelfModelSnapshot:
        snap = SelfModelSnapshot(
            snapshot_id=f"snap-{uuid.uuid4()}",
            at=time.time(),
            state=dict(state),
        )
        with self._lock:
            self._snapshots.append(snap)
        return snap

    def all(self) -> list[SelfModelSnapshot]:
        with self._lock:
            return list(self._snapshots)


class ContradictionDetector:
    def __init__(self, *, ledger: IdentityLedger):
        self.ledger = ledger

    def detect(self, *, candidate_statement: str) -> list[str]:
        # Lightweight contradiction surface: look for negations of fulfilled
        # commitments. Real impl would use a semantic classifier; this is the
        # callable contract.
        lower = candidate_statement.lower().strip()
        contradictions: list[str] = []
        for c in self.ledger.commitments.all():
            if c.fulfilled_at is None and c.revoked_at is None:
                if (
                    f"won't {c.text.lower()}" in lower
                    or f"will not {c.text.lower()}" in lower
                ):
                    contradictions.append(c.commitment_id)
        return contradictions


class IdentityDriftMonitor:
    def __init__(self, *, versioning: SelfModelVersioning):
        self.versioning = versioning

    def drift_score(self) -> float:
        snaps = self.versioning.all()
        if len(snaps) < 2:
            return 0.0
        last = snaps[-1].state
        prev = snaps[-2].state
        keys = set(last.keys()) | set(prev.keys())
        if not keys:
            return 0.0
        diffs = sum(1 for k in keys if last.get(k) != prev.get(k))
        return diffs / len(keys)


class IdentityLedger:
    SCHEMA_VERSION = 1

    def __init__(self, *, root: Path | None = None):
        self.root = Path(root) if root else (state_root() / "identity")
        self.root.mkdir(parents=True, exist_ok=True)
        self.commitments = CommitmentTracker()
        self.preferences = PreferenceHistory()
        self.versioning = SelfModelVersioning()
        self.contradictions = ContradictionDetector(ledger=self)
        self.drift = IdentityDriftMonitor(versioning=self.versioning)

    def persist(self) -> None:
        path = self.root / "identity_ledger.json"
        payload = {
            "commitments": [asdict(c) for c in self.commitments.all()],
            "preferences": {
                "current": dict(self.preferences._current),
                "history": [asdict(c) for c in self.preferences.all_changes()],
            },
            "snapshots": [asdict(s) for s in self.versioning.all()],
            "saved_at": time.time(),
        }
        atomic_write_json(
            path, payload,
            schema_version=self.SCHEMA_VERSION,
            schema_name="identity_ledger",
            # Commitments, preference history and identity snapshots cannot be
            # reconstructed from anything else if the machine loses power, and
            # a lost promise is not a lost cache entry. On macOS plain fsync
            # hands the write to the drive's cache and returns, so only
            # F_FULLFSYNC actually makes this survive. Measured cost on this
            # host: ~8ms against ~0.2ms, paid once per ledger save.
            power_safe=True,
        )

    def load(self) -> None:
        path = self.root / "identity_ledger.json"
        if not path.exists():
            return
        env = read_json_envelope(path)
        payload = env.get("payload") or {}
        for c in payload.get("commitments", []):
            commitment = Commitment(**c)
            self.commitments._commitments[commitment.commitment_id] = commitment
        prefs = payload.get("preferences") or {}
        self.preferences._current = dict(prefs.get("current") or {})
        for h in prefs.get("history", []):
            self.preferences._history.append(PreferenceChange(**h))
        for s in payload.get("snapshots", []):
            self.versioning._snapshots.append(SelfModelSnapshot(**s))


_global: IdentityLedger | None = None


def get_identity_ledger() -> IdentityLedger:
    global _global
    if _global is None:
        _global = IdentityLedger()
    return _global


def reset_identity_ledger() -> None:
    global _global
    _global = None
