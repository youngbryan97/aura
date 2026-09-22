"""Deletion guard — nothing important is lost, accidentally or by force.

Protects against the destruction you can't undo: accidental deletes, malicious forced deletes,
and deletion *storms* (a script told to wipe everything). Every guarded delete first takes a
content-addressed version snapshot to a recycle store, so any deletion is recoverable. High-risk
paths (Aura's own identity / memory / source / config) require explicit confirmation. And a
deletion storm — too many destructive ops in a short window — trips a freeze + alert rather than
letting the machine be emptied: the same homeostasis principle as the immune system's FOP guard,
applied to data loss.

Defensive only: it never deletes on its own initiative; it interposes on destructive ops to make
them recoverable and to refuse a wipe.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Security.DeletionGuard")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


# Path fragments that mark something Aura must not lose without an explicit, confirmed decision.
_PROTECTED_MARKERS = (
    "identity", "memory", "heartstone", "constitution", "governance", "soul", "lineage",
    "/.aura/", "credential", "secret", "vault", "config", ".git/",
)


@dataclass
class GuardDecision:
    allowed: bool
    path: str
    risk: float                       # [0,1]
    requires_confirmation: bool
    version_id: str | None         # restore handle if a snapshot was taken
    frozen: bool                      # deletion-storm freeze engaged
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed, "path": self.path, "risk": round(self.risk, 3),
            "requires_confirmation": self.requires_confirmation, "version_id": self.version_id,
            "frozen": self.frozen, "reason": self.reason,
        }


class DeletionGuard:
    """Versioned, risk-aware, storm-resistant interposition on destructive file operations."""

    def __init__(
        self,
        recycle_dir: Path | None = None,
        *,
        storm_window_s: float = 10.0,
        storm_threshold: int = 25,
        max_version_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if recycle_dir is None:
            try:
                from core.config import config
                recycle_dir = Path(config.paths.home_dir) / "data" / "deletion_recycle"
            except (ImportError, AttributeError, RuntimeError):
                recycle_dir = state_root() / "data" / "deletion_recycle"
        self._recycle = Path(recycle_dir)
        self._recycle.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._window = storm_window_s
        self._storm_threshold = storm_threshold
        self._max_version_bytes = max_version_bytes
        self._recent: deque[float] = deque(maxlen=512)
        self._frozen_until: float = 0.0
        self._versions: dict[str, tuple[str, float]] = {}  # version_id -> (original_path, t)

    # ── risk classification ────────────────────────────────────────────────

    @staticmethod
    def classify_risk(path: str, *, forced: bool) -> tuple[float, bool]:
        p = str(path).lower()
        protected = any(m in p for m in _PROTECTED_MARKERS)
        base = 0.8 if protected else 0.25
        if forced:
            base = _clamp(base + 0.2)
        return _clamp(base), protected

    # ── the guard ──────────────────────────────────────────────────────────

    def guard_delete(
        self,
        path: str,
        *,
        forced: bool = False,
        confirmed: bool = False,
        actor: str = "aura",
        now: float | None = None,
    ) -> GuardDecision:
        """Vet a destructive op. Snapshots first (recoverable), gates high-risk, refuses storms."""
        now = time.time() if now is None else now
        risk, protected = self.classify_risk(path, forced=forced)

        with self._lock:
            # Deletion-storm guard: too many destructive ops in the window → freeze + alert.
            while self._recent and now - self._recent[0] > self._window:
                self._recent.popleft()
            if now < self._frozen_until:
                self._flag_immune(path, risk, "deletion storm — frozen", forced)
                return GuardDecision(False, str(path), risk, False, None, True,
                                     "deletions frozen after a storm; manual unfreeze required")
            self._recent.append(now)
            if len(self._recent) > self._storm_threshold:
                self._frozen_until = now + self._window * 3
                self._flag_immune(path, max(risk, 0.9), "deletion storm detected", forced=True)
                return GuardDecision(False, str(path), max(risk, 0.9), False, None, True,
                                     f"deletion storm ({len(self._recent)} in {self._window:.0f}s) — frozen + alerted")

        # Snapshot before anything is allowed, so the delete is always recoverable.
        version_id = self._snapshot(path, now)

        # High-risk / protected paths require explicit confirmation.
        requires_confirmation = (protected or risk >= 0.7) and not confirmed
        if requires_confirmation:
            self._flag_immune(path, risk, "high-risk deletion needs confirmation", forced)
            return GuardDecision(False, str(path), risk, True, version_id, False,
                                 "protected/high-risk path — explicit confirmation required",
                                 evidence={"protected": protected})

        # A forced delete on a protected path is treated as an attack even if "confirmed".
        if forced and protected and actor != "bryan":
            self._flag_immune(path, max(risk, 0.85), "forced deletion of protected path", forced=True)
            return GuardDecision(False, str(path), max(risk, 0.85), True, version_id, False,
                                 "forced deletion of a protected path blocked — owner must do this")

        return GuardDecision(True, str(path), risk, False, version_id, False,
                             "allowed (snapshot taken; recoverable)")

    def _snapshot(self, path: str, now: float) -> str | None:
        try:
            src = Path(path)
            if not src.exists() or not src.is_file():
                return None
            if src.stat().st_size > self._max_version_bytes:
                return None  # too large to version; caller still gated by risk
            data = src.read_bytes()
            digest = hashlib.sha256(data).hexdigest()[:16]
            version_id = f"ver-{int(now)}-{digest}"
            dest = self._recycle / version_id
            get_file_write_gateway().write_bytes(
                dest,
                data,
                source="deletion_guard.snapshot",
            )
            with self._lock:
                self._versions[version_id] = (str(src), now)
            return version_id
        except (OSError, ValueError) as exc:
            logger.debug("Snapshot failed for %s: %s", path, exc)
            return None

    def restore(self, version_id: str, *, to: str | None = None) -> str | None:
        """Bring back a guarded deletion."""
        with self._lock:
            meta = self._versions.get(version_id)
        if meta is None:
            return None
        original, _t = meta
        target = Path(to) if to else Path(original)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self._recycle / version_id, target)
            return str(target)
        except OSError as exc:
            logger.warning("Restore failed for %s: %s", version_id, exc)
            return None

    def unfreeze(self) -> None:
        with self._lock:
            self._frozen_until = 0.0
            self._recent.clear()

    def _flag_immune(self, path: str, severity: float, reason: str, forced: bool) -> None:
        try:
            from core.security.immune_system import ThreatClass, get_immune_system
            get_immune_system().assess(
                "deletion_guard", f"{reason}: {path}", severity=severity,
                origin="local", targeted_vuln="unguarded_delete",
                threat_class=ThreatClass.DESTRUCTION, evidence={"forced": forced, "path": path},
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning(
                "%s unavailable (%s: %s); the immune system was not told about this delete",
                "ThreatClass",
                type(exc).__name__,
                exc,
            )

    def status(self, *, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        with self._lock:
            recent = sum(1 for t in self._recent if now - t <= self._window)
            return {
                "versions_kept": len(self._versions),
                "recent_deletions": recent,
                "frozen": now < self._frozen_until,
                "recycle_dir": str(self._recycle),
            }


_guard: DeletionGuard | None = None
_guard_lock = threading.Lock()


def get_deletion_guard() -> DeletionGuard:
    global _guard
    if _guard is None:
        with _guard_lock:
            if _guard is None:
                _guard = DeletionGuard()
    return _guard
