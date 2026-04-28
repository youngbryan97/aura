"""Universal receipt types and durable receipt store.

The audit insists every consequential action emits a receipt and that the
chain cause -> decision -> action -> result is forensically reconstructible.
This module defines the ten canonical receipt types and a `ReceiptStore`
that persists them through the canonical AtomicWriter so every receipt is
durable, schema-versioned, and queryable.
"""
from __future__ import annotations


import json
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.runtime.atomic_writer import atomic_write_json, read_json_envelope


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"


@dataclass
class _ReceiptBase:
    # All base fields default-able so subclasses may safely add their own
    # default-bearing fields without violating dataclass field ordering.
    receipt_id: str = ""
    kind: str = ""
    cause: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TurnReceipt(_ReceiptBase):
    kind: str = "turn"
    origin: str = ""
    governance_receipt_id: Optional[str] = None
    committed_effects: List[str] = field(default_factory=list)
    failed_effects: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class GovernanceReceipt(_ReceiptBase):
    kind: str = "governance"
    domain: str = ""
    action: str = ""
    approved: bool = False
    reason: str = ""


@dataclass
class CapabilityReceipt(_ReceiptBase):
    kind: str = "capability"
    capability: str = ""
    scope: str = ""
    issuer: str = "UnifiedWill"
    expires_at: float = 0.0
    revoked: bool = False


@dataclass
class ToolExecutionReceipt(_ReceiptBase):
    kind: str = "tool_execution"
    tool: str = ""
    governance_receipt_id: Optional[str] = None
    capability_receipt_id: Optional[str] = None
    status: str = "success_unverified"
    output_digest: Optional[str] = None
    verification_evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryWriteReceipt(_ReceiptBase):
    kind: str = "memory_write"
    family: str = ""
    record_id: str = ""
    bytes_written: int = 0
    schema_version: int = 1
    governance_receipt_id: Optional[str] = None


@dataclass
class StateMutationReceipt(_ReceiptBase):
    kind: str = "state_mutation"
    domain: str = ""
    key: str = ""
    schema_version: int = 1
    governance_receipt_id: Optional[str] = None


@dataclass
class OutputReceipt(_ReceiptBase):
    kind: str = "output"
    origin: str = ""
    target: str = ""
    digest: str = ""
    governance_receipt_id: Optional[str] = None


@dataclass
class AutonomyReceipt(_ReceiptBase):
    kind: str = "autonomy"
    autonomy_level: int = 0
    proposed_action: str = ""
    governance_receipt_id: Optional[str] = None
    budget_remaining: float = 0.0


@dataclass
class SelfRepairReceipt(_ReceiptBase):
    kind: str = "self_repair"
    target_module: str = ""
    rungs_passed: List[str] = field(default_factory=list)
    rolled_back: bool = False
    governance_receipt_id: Optional[str] = None


@dataclass
class ComputerUseReceipt(_ReceiptBase):
    kind: str = "computer_use"
    action_kind: str = ""
    target: str = ""
    screen_before_hash: str = ""
    screen_after_hash: str = ""
    capability_receipt_id: Optional[str] = None
    verifier_result: bool = False


# Mapping kind -> dataclass for the store.
_RECEIPT_CLASSES = {
    "turn": TurnReceipt,
    "governance": GovernanceReceipt,
    "capability": CapabilityReceipt,
    "tool_execution": ToolExecutionReceipt,
    "memory_write": MemoryWriteReceipt,
    "state_mutation": StateMutationReceipt,
    "output": OutputReceipt,
    "autonomy": AutonomyReceipt,
    "self_repair": SelfRepairReceipt,
    "computer_use": ComputerUseReceipt,
}


AnyReceipt = Union[
    TurnReceipt,
    GovernanceReceipt,
    CapabilityReceipt,
    ToolExecutionReceipt,
    MemoryWriteReceipt,
    StateMutationReceipt,
    OutputReceipt,
    AutonomyReceipt,
    SelfRepairReceipt,
    ComputerUseReceipt,
]


class ReceiptStore:
    """Durable receipt store backed by the canonical AtomicWriter.

    Every receipt is written as a single schema-versioned JSON envelope
    in ``root/<kind>/<receipt_id>.json``. The store also maintains an
    in-memory index for fast querying within a process; on cold restart
    the index can be rebuilt from disk.
    """

    SCHEMA_VERSION = 1

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root is not None else (Path.home() / ".aura" / "receipts")
        get_task_tracker().create_task(get_storage_gateway().create_dir(self.root, cause='ReceiptStore.__init__'))
        self._lock = threading.RLock()
        self._index: Dict[str, AnyReceipt] = {}

    def emit(self, receipt: AnyReceipt) -> AnyReceipt:
        if not getattr(receipt, "receipt_id", None):
            receipt.receipt_id = _new_id(receipt.kind)
        path = self.root / receipt.kind / f"{receipt.receipt_id}.json"
        atomic_write_json(
            path,
            receipt.to_dict(),
            schema_version=self.SCHEMA_VERSION,
            schema_name=f"receipt_{receipt.kind}",
        )
        with self._lock:
            self._index[receipt.receipt_id] = receipt
        return receipt

    def get(self, receipt_id: str) -> Optional[AnyReceipt]:
        with self._lock:
            return self._index.get(receipt_id)

    def query_by_kind(self, kind: str) -> List[AnyReceipt]:
        if kind not in _RECEIPT_CLASSES:
            raise ValueError(f"unknown receipt kind '{kind}'")
        with self._lock:
            return [r for r in self._index.values() if r.kind == kind]

    def query_recent(
        self,
        *,
        kinds: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[AnyReceipt]:
        """Return the newest receipts across one or more kinds."""
        with self._lock:
            receipts = list(self._index.values())

        if kinds:
            allowed = {str(kind or "").strip() for kind in kinds if str(kind or "").strip()}
            receipts = [receipt for receipt in receipts if receipt.kind in allowed]

        receipts.sort(key=lambda receipt: float(getattr(receipt, "created_at", 0.0) or 0.0))
        if limit <= 0:
            return []
        return receipts[-limit:]

    def reload_from_disk(self) -> int:
        """Rebuild the in-memory index from disk. Returns count loaded."""
        count = 0
        with self._lock:
            self._index.clear()
            for kind, cls in _RECEIPT_CLASSES.items():
                kind_dir = self.root / kind
                if not kind_dir.exists():
                    continue
                for jf in kind_dir.glob("*.json"):
                    try:
                        env = read_json_envelope(jf)
                        payload = env.get("payload") or {}
                        # Strip kind from payload to avoid passing twice.
                        payload.pop("kind", None)
                        receipt = cls(**payload)
                        receipt.kind = kind
                        self._index[receipt.receipt_id] = receipt
                        count += 1
                    except Exception:
                        continue
        return count

    def coverage_stats(self) -> Dict[str, int]:
        with self._lock:
            stats: Dict[str, int] = {kind: 0 for kind in _RECEIPT_CLASSES}
            for r in self._index.values():
                stats[r.kind] = stats.get(r.kind, 0) + 1
            return stats


_global_store: Optional[ReceiptStore] = None
_singleton_lock = threading.RLock()


def get_receipt_store(root: Optional[Path] = None) -> ReceiptStore:
    global _global_store
    with _singleton_lock:
        if _global_store is None:
            _global_store = ReceiptStore(root)
        return _global_store


def reset_receipt_store() -> None:
    global _global_store
    with _singleton_lock:
        _global_store = None
