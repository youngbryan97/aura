"""Evidence mode — fail-closed controls for hostile review.

The critique's single sharpest demand: in evidence mode, no silent fallback
may be credited as live behavior. If the substrate isn't live, steering
fails. If the vectors weren't derived from real hidden states, steering
fails. No random-vector stand-ins, no neutral-state substitutions, no
"best effort" telemetry claiming success.

Set ``AURA_EVIDENCE_MODE=1`` (or pass ``evidence_mode=True`` to a runtime
service) to activate. The normal runtime stays forgiving; only the evidence
path is strict.
"""
from __future__ import annotations
from core.runtime.atomic_writer import atomic_write_text

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class EvidenceViolation:
    kind: str
    detail: str
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail, "timestamp": self.timestamp}


class EvidenceMode:
    """Thread-safe evidence-mode policy.

    Public API is deliberately small: ``active()`` tells callers whether
    fail-closed rules apply, ``require_or_fail()`` is used by the code paths
    the critique named (steering vectors, substrate sync, hook installation,
    evidence hooks), and ``violations()`` exposes an audit trail so a
    reviewer can see exactly where a run would have cheated.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._override: Optional[bool] = None
        self._violations: List[EvidenceViolation] = []

    # ------------------------------------------------------------------
    def active(self) -> bool:
        with self._lock:
            if self._override is not None:
                return bool(self._override)
        return _env_truthy(os.environ.get("AURA_EVIDENCE_MODE"))

    def set_override(self, value: Optional[bool]) -> None:
        with self._lock:
            self._override = None if value is None else bool(value)

    def clear_violations(self) -> None:
        with self._lock:
            self._violations.clear()

    def violations(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [v.as_dict() for v in self._violations]

    # ------------------------------------------------------------------
    def require_or_fail(
        self,
        kind: str,
        condition: bool,
        detail: str,
        *,
        raise_cls: type[Exception] = RuntimeError,
    ) -> None:
        """Raise if evidence mode is active and the condition fails."""
        if condition:
            return
        violation = EvidenceViolation(kind=kind, detail=detail)
        with self._lock:
            self._violations.append(violation)
        if self.active():
            raise raise_cls(f"[evidence_mode] {kind}: {detail}")

    # ------------------------------------------------------------------
    def record(self, kind: str, detail: str) -> None:
        with self._lock:
            self._violations.append(EvidenceViolation(kind=kind, detail=detail))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "active": self.active(),
            "violations": self.violations(),
        }

    def dump(self, path: str | Path) -> None:
        p = Path(path)
        get_task_tracker().create_task(get_storage_gateway().create_dir(p.parent, cause='EvidenceMode.dump'))
        atomic_write_text(p, json.dumps(self.snapshot(), indent=2, sort_keys=True) + "\n")


_singleton: Optional[EvidenceMode] = None
_lock = threading.Lock()


def get_evidence_mode() -> EvidenceMode:
    global _singleton
    with _lock:
        if _singleton is None:
            _singleton = EvidenceMode()
        return _singleton


def reset_singleton_for_test() -> None:
    global _singleton
    with _lock:
        _singleton = None


def _env_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# Convenience wrapper for frequent callers ---------------------------------
def require(kind: str, condition: bool, detail: str) -> None:
    get_evidence_mode().require_or_fail(kind, condition, detail)


def record_violation(kind: str, detail: str) -> None:
    get_evidence_mode().record(kind, detail)


def is_evidence_mode() -> bool:
    return get_evidence_mode().active()
