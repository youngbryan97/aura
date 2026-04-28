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
from core.runtime.audit_chain import AuditChain, ChainEntry


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


@dataclass
class SemanticWeightUpdateReceipt(_ReceiptBase):
    """Forensic record of a plastic-adapter weight update.

    Emitted by the grounding loop after a prediction is confirmed and
    the governor has authorised the resulting Hebbian update.  Fields
    record exactly which module was modified, why, and how much, so an
    auditor can reconstruct every weight change without inspecting the
    live arrays.
    """

    kind: str = "semantic_weight_update"
    module: str = ""
    prediction_id: Optional[str] = None
    concept_id: Optional[str] = None
    evidence_id: Optional[str] = None
    reward: float = 0.0
    modulation: float = 0.0
    delta_norm: float = 0.0
    hebb_norm: float = 0.0
    allowed: bool = False
    governance_receipt_id: Optional[str] = None


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
    "semantic_weight_update": SemanticWeightUpdateReceipt,
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
    SemanticWeightUpdateReceipt,
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
        # Schedule durable dir creation through the storage gateway when it
        # is available (production runtime); fall back to a synchronous
        # mkdir so unit tests and bootstrap paths still work.  The mkdir
        # is idempotent and load-bearing for the audit chain sidecar.
        try:
            get_task_tracker().create_task(  # type: ignore[name-defined]
                get_storage_gateway().create_dir(  # type: ignore[name-defined]
                    self.root, cause='ReceiptStore.__init__'
                )
            )
        except NameError:
            pass
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._index: Dict[str, AnyReceipt] = {}
        # Tamper-evident chain lives at root/_chain.jsonl. Sidecar; do not
        # break existing callers if the chain file cannot be initialised.
        self._chain: Optional[AuditChain] = None
        try:
            self._chain = AuditChain(self.root)
        except Exception:
            self._chain = None

    def emit(self, receipt: AnyReceipt) -> AnyReceipt:
        if not getattr(receipt, "receipt_id", None):
            receipt.receipt_id = _new_id(receipt.kind)
        path = self.root / receipt.kind / f"{receipt.receipt_id}.json"
        body = receipt.to_dict()
        atomic_write_json(
            path,
            body,
            schema_version=self.SCHEMA_VERSION,
            schema_name=f"receipt_{receipt.kind}",
        )
        with self._lock:
            self._index[receipt.receipt_id] = receipt
        # Append to tamper-evident chain after the receipt is durable on
        # disk so verifiers always find a body to re-hash.
        if self._chain is not None:
            try:
                self._chain.append(
                    receipt_id=receipt.receipt_id,
                    kind=receipt.kind,
                    body=body,
                    timestamp=float(getattr(receipt, "created_at", 0.0) or 0.0),
                )
            except Exception:
                # Chain failure must not bring down the emit path; the
                # receipt is already durable.  A subsequent verify() will
                # surface the gap.
                pass
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

    def _load_body_from_disk(self, receipt_id: str, kind: str) -> Optional[Dict[str, Any]]:
        """Re-read a receipt body from disk for chain verification."""
        path = self.root / kind / f"{receipt_id}.json"
        if not path.exists():
            return None
        try:
            env = read_json_envelope(path)
        except Exception:
            return None
        payload = env.get("payload") if isinstance(env, dict) else None
        if not isinstance(payload, dict):
            return None
        # Re-attach kind so the body matches what was hashed at emit-time.
        payload.setdefault("kind", kind)
        return payload

    def verify_chain(self) -> Dict[str, Any]:
        """Verify the tamper-evident chain.

        Returns a dict with ``ok`` (bool), ``length`` (int),
        ``head_hash`` (str), and ``problems`` (list).  ``problems`` is
        empty when verification passes.
        """
        if self._chain is None:
            return {"ok": False, "length": 0, "head_hash": "", "problems": [
                {"reason": "chain not initialised"}
            ]}
        ok, problems = self._chain.verify(body_loader=self._load_body_from_disk)
        return {
            "ok": ok,
            "length": self._chain.length(),
            "head_hash": self._chain.head_hash(),
            "problems": problems,
        }

    def export_chain(self, dest_dir: Path) -> Dict[str, Any]:
        """Export the chain (chain.jsonl + MANIFEST.txt) to ``dest_dir``."""
        if self._chain is None:
            raise RuntimeError("chain not initialised")
        return self._chain.export(dest_dir)


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
