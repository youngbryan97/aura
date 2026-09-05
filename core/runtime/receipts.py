"""Universal receipt types and durable receipt store.

The audit insists every consequential action emits a receipt and that the
chain cause -> decision -> action -> result is forensically reconstructible.
This module defines the canonical receipt types and a `ReceiptStore`
that persists them through the canonical AtomicWriter so every receipt is
durable, schema-versioned, and queryable.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import contextlib
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from core.runtime.atomic_writer import (
    atomic_append_text,
    atomic_write_json,
    atomic_write_text,
    read_json_envelope,
)
from core.runtime.audit_chain import AuditChain
from core.runtime.flags import FlagKind, declare
from core.runtime.state_ownership import state_root
from core.runtime.store_locality import assert_wal_safe
from core.runtime.sqlite_support import connection_is_open, open_tracked

logger = logging.getLogger("core.runtime.receipts")

_HIGH_VOLUME_RECEIPT_KINDS = frozenset(
    {"autonomy", "output", "resource_admission", "workspace_gate"}
)
_USER_FACING_OUTPUT_SINKS = frozenset(
    {
        "event_bus",
        "http_response_body",
        "multimodal_renderer",
        "mycelial_ui",
        "reply_queue",
        "voice_engine",
    }
)
_HOT_INDEX_LIMIT_FLAG = declare(
    "AURA_RECEIPT_HOT_INDEX_LIMIT",
    kind=FlagKind.INT,
    default=2048,
    description="Maximum receipts retained per kind in the process hot index",
    owner="core.runtime.receipts",
)
_RECEIPT_ROOT_FLAG = declare(
    "AURA_RECEIPT_ROOT",
    kind=FlagKind.STRING,
    default="",
    description="Override the durable receipt-store root for isolated runtimes and tests",
    owner="core.runtime.receipts",
)


class PracticeCurriculumStore:
    """Bounded receipt-pinned curriculum ledger with one filesystem owner.

    Practice observations retain their source receipt identifiers, while this
    store owns directory creation, append/compaction, and the seen-index
    replacement. Callers cannot choose arbitrary output paths.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.ledger_path = self.root / "practice_curriculum.jsonl"
        self.seen_path = self.root / "practice_curriculum_seen.json"

    def persist(
        self,
        *,
        ledger_body: str,
        seen_json: str,
        full_rewrite: bool,
    ) -> None:
        if not isinstance(ledger_body, str) or not isinstance(seen_json, str):
            raise TypeError("practice curriculum payloads must be text")
        from core.governance_context import governance_runtime_active, require_governance

        if governance_runtime_active():
            require_governance(
                "practice_curriculum_store.persist",
                strict=True,
                allowed_domains=("memory_write",),
            )
        if full_rewrite:
            atomic_write_text(self.ledger_path, ledger_body)
        elif ledger_body:
            atomic_append_text(self.ledger_path, ledger_body)
        atomic_write_text(self.seen_path, seen_json)


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
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TurnReceipt(_ReceiptBase):
    kind: str = "turn"
    origin: str = ""
    governance_receipt_id: str | None = None
    committed_effects: list[str] = field(default_factory=list)
    failed_effects: list[dict[str, str]] = field(default_factory=list)


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
    governance_receipt_id: str | None = None
    capability_receipt_id: str | None = None
    status: str = "success_unverified"
    output_digest: str | None = None
    verification_evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryWriteReceipt(_ReceiptBase):
    kind: str = "memory_write"
    family: str = ""
    record_id: str = ""
    bytes_written: int = 0
    schema_version: int = 1
    governance_receipt_id: str | None = None


@dataclass
class StateMutationReceipt(_ReceiptBase):
    kind: str = "state_mutation"
    domain: str = ""
    key: str = ""
    schema_version: int = 1
    governance_receipt_id: str | None = None


@dataclass
class OutputReceipt(_ReceiptBase):
    kind: str = "output"
    origin: str = ""
    target: str = ""
    digest: str = ""
    governance_receipt_id: str | None = None


def digest_output_content(content: Any) -> str:
    """Return the canonical bounded digest for an exact delivered response."""
    return hashlib.sha256(
        str(content or "").encode("utf-8", errors="replace")
    ).hexdigest()[:16]


def digest_principal_binding(principal: Any) -> str:
    """Return a non-reversible binding for one exact authenticated principal."""
    exact_principal = str(principal or "")
    if not exact_principal:
        return ""
    try:
        encoded = exact_principal.encode("utf-8", errors="strict")
    except UnicodeError:
        return ""
    return hashlib.sha256(encoded).hexdigest()


def validate_transport_output_receipt(
    receipt: object,
    *,
    content: Any,
    principal: Any,
) -> bool:
    """Fail closed unless a durable output proof matches content and principal."""
    if not isinstance(receipt, OutputReceipt) or not receipt.receipt_id:
        return False
    if receipt.kind != "output" or receipt.target not in {"primary", "both"}:
        return False
    metadata = receipt.metadata
    if not isinstance(metadata, dict):
        return False
    accepted_sinks = metadata.get("accepted_sinks")
    if (
        not isinstance(accepted_sinks, list)
        or not accepted_sinks
        or any(
            not isinstance(sink, str) or sink not in _USER_FACING_OUTPUT_SINKS
            for sink in accepted_sinks
        )
    ):
        return False
    if metadata.get("delivery_stage") != "transport_accepted":
        return False
    expected_content_digest = digest_output_content(content)
    expected_principal_digest = digest_principal_binding(principal)
    actual_content_digest = receipt.digest
    actual_principal_digest = metadata.get("recipient_principal_digest")
    if (
        not expected_principal_digest
        or not isinstance(actual_content_digest, str)
        or not isinstance(actual_principal_digest, str)
    ):
        return False
    return hmac.compare_digest(
        actual_content_digest,
        expected_content_digest,
    ) and hmac.compare_digest(
        actual_principal_digest,
        expected_principal_digest,
    )


@dataclass
class AutonomyReceipt(_ReceiptBase):
    kind: str = "autonomy"
    autonomy_level: int = 0
    proposed_action: str = ""
    governance_receipt_id: str | None = None
    budget_remaining: float = 0.0


@dataclass
class SelfRepairReceipt(_ReceiptBase):
    kind: str = "self_repair"
    target_module: str = ""
    rungs_passed: list[str] = field(default_factory=list)
    rolled_back: bool = False
    governance_receipt_id: str | None = None


@dataclass
class ComputerUseReceipt(_ReceiptBase):
    kind: str = "computer_use"
    action_kind: str = ""
    target: str = ""
    screen_before_hash: str = ""
    screen_after_hash: str = ""
    capability_receipt_id: str | None = None
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
    prediction_id: str | None = None
    concept_id: str | None = None
    evidence_id: str | None = None
    reward: float = 0.0
    modulation: float = 0.0
    delta_norm: float = 0.0
    hebb_norm: float = 0.0
    allowed: bool = False
    governance_receipt_id: str | None = None


@dataclass
class DegradationReceipt(_ReceiptBase):
    """Forensic record of a recorded degradation (see core/runtime/errors.py).

    Registered here so the bounded store can snapshot and reconstruct it
    like every other kind — an unregistered kind makes emit() raise, and
    record_degradation is called from exception handlers everywhere.
    """

    kind: str = "degradation"
    subsystem: str = ""
    severity_level: str = ""
    error_type_name: str = ""
    error_message_text: str = ""
    action_taken: str = ""
    extra_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceAdmissionReceipt(_ReceiptBase):
    """Durable decision record for constrained runtime work admission."""

    kind: str = "resource_admission"
    request_id: str = ""
    owner: str = ""
    work_class: str = ""
    lane: str = ""
    priority: int = 0
    decision: str = ""
    reason: str = ""
    lease_id: str = ""
    pressure: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkspaceGateReceipt(_ReceiptBase):
    """Durable fail-closed decision from the cognitive workspace authority gate."""

    kind: str = "workspace_gate"
    candidate_source: str = ""
    gate: str = "global_inhibition"
    decision: str = "rejected"
    reason: str = ""
    retryable: bool = True
    gate_instance_id: str = ""


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
    "resource_admission": ResourceAdmissionReceipt,
    "workspace_gate": WorkspaceGateReceipt,
    "degradation": DegradationReceipt,
}


AnyReceipt = (
    TurnReceipt
    | GovernanceReceipt
    | CapabilityReceipt
    | ToolExecutionReceipt
    | MemoryWriteReceipt
    | StateMutationReceipt
    | OutputReceipt
    | AutonomyReceipt
    | SelfRepairReceipt
    | ComputerUseReceipt
    | SemanticWeightUpdateReceipt
    | ResourceAdmissionReceipt
    | WorkspaceGateReceipt
    | DegradationReceipt
)


class ReceiptStore:
    """Durable receipt store backed by the canonical AtomicWriter.

    Ordinary receipts are written as schema-versioned JSON envelopes in
    ``root/<kind>/<receipt_id>.json``. High-volume operational receipts use a
    WAL-backed SQLite ledger so long-running runtimes do not create an unbounded
    inode count. Both formats feed the same tamper-evident audit chain and API.
    """

    SCHEMA_VERSION = 1

    def __init__(self, root: Path | None = None):
        configured_root = str(_RECEIPT_ROOT_FLAG.value() or "").strip()
        self.root = (
            Path(root)
            if root is not None
            else Path(configured_root)
            if configured_root
            else (state_root() / "receipts")
        )
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self.root.chmod(0o700)
        except OSError as exc:
            logger.debug("Could not restrict receipt root permissions: %s", exc)
        self._lock = threading.RLock()
        self._closed = False
        self._index: dict[str, AnyReceipt] = {}
        self._chain_append_errors: list[dict[str, Any]] = []
        self._ledger_path = self.root / "_high_volume_receipts.sqlite3"
        self._ledger: sqlite3.Connection | None = None
        self._ledger_pid = 0
        self._ledger_available = self._initialize_high_volume_ledger()
        # Tamper-evident chain lives at root/_chain.jsonl. Sidecar; do not
        # break existing callers if the chain file cannot be initialised.
        self._chain: AuditChain | None = None
        try:
            self._chain = AuditChain(self.root)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            self._chain = None

    @property
    def hot_index_limit(self) -> int:
        return max(64, int(_HOT_INDEX_LIMIT_FLAG.value()))

    def _ledger_connection_locked(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("receipt store is closed")
        current_pid = os.getpid()
        if (
            self._ledger is not None
            and self._ledger_pid == current_pid
            and connection_is_open(self._ledger)
        ):
            return self._ledger
        if self._ledger is not None:
            try:
                self._ledger.close()
            except sqlite3.Error:
                pass
        # WAL's shared-memory index is a same-machine mechanism. On a network
        # filesystem it is not coherent between hosts and the failure mode is
        # silent corruption, not an error — see core/runtime/store_locality.py,
        # which also carries the measurement showing that WAL does NOT impose
        # the single-PROCESS lock it was assumed to.
        assert_wal_safe(self._ledger_path, subsystem="receipts.ledger")
        connection = open_tracked(
            self._ledger_path,
            timeout=5.0,
            check_same_thread=False,
        )
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=5000")
        self._ledger = connection
        self._ledger_pid = current_pid
        return connection

    def _initialize_high_volume_ledger(self) -> bool:
        try:
            with self._lock:
                connection = self._ledger_connection_locked()
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS receipt_ledger (
                        receipt_id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        body_json TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_receipt_ledger_kind_created
                    ON receipt_ledger(kind, created_at DESC)
                    """
                )
                connection.commit()
                try:
                    self._ledger_path.chmod(0o600)
                except OSError as exc:
                    logger.debug("Could not restrict receipt ledger permissions: %s", exc)
            return True
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            if self._ledger is not None:
                try:
                    self._ledger.close()
                except sqlite3.Error:
                    pass
                self._ledger = None
                self._ledger_pid = 0
            logger.error(
                "High-volume receipt ledger unavailable; falling back to envelope files: %s",
                exc,
            )
            return False

    def _ledger_put_locked(self, body: dict[str, Any]) -> None:
        connection = self._ledger_connection_locked()
        receipt_id = str(body.get("receipt_id") or "")
        kind = str(body.get("kind") or "")
        body_json = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        try:
            connection.execute(
                """
                INSERT INTO receipt_ledger(receipt_id, kind, created_at, body_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    receipt_id,
                    kind,
                    float(body.get("created_at") or 0.0),
                    body_json,
                ),
            )
        except sqlite3.IntegrityError as exc:
            row = connection.execute(
                "SELECT kind, body_json FROM receipt_ledger WHERE receipt_id = ?",
                (receipt_id,),
            ).fetchone()
            if row is None or str(row[0]) != kind or str(row[1]) != body_json:
                raise ValueError(f"receipt id is immutable and already exists: {receipt_id}") from exc
        connection.commit()

    def _ledger_body_locked(self, receipt_id: str, kind: str) -> dict[str, Any] | None:
        if not self._ledger_available:
            return None
        try:
            row = self._ledger_connection_locked().execute(
                "SELECT body_json FROM receipt_ledger WHERE receipt_id = ? AND kind = ?",
                (str(receipt_id), str(kind)),
            ).fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        try:
            body = json.loads(str(row[0]))
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        return body if isinstance(body, dict) else None

    @staticmethod
    def _receipt_from_body(kind: str, body: dict[str, Any]) -> AnyReceipt | None:
        cls = _RECEIPT_CLASSES.get(kind)
        if cls is None:
            return None
        payload = dict(body)
        payload.pop("kind", None)
        try:
            receipt = cls(**payload)
        except TypeError:
            return None
        receipt.kind = kind
        return cast(AnyReceipt, receipt)

    @classmethod
    def _receipt_snapshot(cls, receipt: AnyReceipt) -> AnyReceipt:
        snapshot = cls._receipt_from_body(receipt.kind, receipt.to_dict())
        if snapshot is None:
            raise ValueError(f"receipt cannot be reconstructed: {receipt.kind}")
        return snapshot

    def _prune_hot_index_locked(self, kind: str) -> None:
        receipts = sorted(
            (
                receipt
                for receipt in self._index.values()
                if receipt.kind == kind
            ),
            key=lambda receipt: float(getattr(receipt, "created_at", 0.0) or 0.0),
            reverse=True,
        )
        for receipt in receipts[self.hot_index_limit :]:
            self._index.pop(receipt.receipt_id, None)

    def emit(self, receipt: AnyReceipt) -> AnyReceipt:
        if self._closed:
            raise RuntimeError("receipt store is closed")
        if not getattr(receipt, "receipt_id", None):
            receipt.receipt_id = _new_id(receipt.kind)
        body = receipt.to_dict()
        with self._lock:
            existing = self.get(receipt.receipt_id)
            if existing is not None:
                if existing.kind != receipt.kind or existing.to_dict() != body:
                    raise ValueError(
                        f"receipt id is immutable and already exists: {receipt.receipt_id}"
                    )
                return self._receipt_snapshot(existing)
            if receipt.kind in _HIGH_VOLUME_RECEIPT_KINDS and self._ledger_available:
                try:
                    self._ledger_put_locked(body)
                except (OSError, sqlite3.Error) as exc:
                    logger.error(
                        "High-volume receipt ledger write failed; using durable envelope fallback: %s",
                        exc,
                    )
                    if self._ledger is not None:
                        try:
                            self._ledger.close()
                        except sqlite3.Error:
                            pass
                    self._ledger = None
                    self._ledger_pid = 0
                    self._ledger_available = False
                    path = self.root / receipt.kind / f"{receipt.receipt_id}.json"
                    atomic_write_json(
                        path,
                        body,
                        schema_version=self.SCHEMA_VERSION,
                        schema_name=f"receipt_{receipt.kind}",
                        indent=None,
                    )
            else:
                path = self.root / receipt.kind / f"{receipt.receipt_id}.json"
                atomic_write_json(
                    path,
                    body,
                    schema_version=self.SCHEMA_VERSION,
                    schema_name=f"receipt_{receipt.kind}",
                    indent=None,
                )
            self._index[receipt.receipt_id] = self._receipt_snapshot(receipt)
            self._prune_hot_index_locked(receipt.kind)
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
            except (RuntimeError, AttributeError, TypeError) as exc:
                # Chain failure must not bring down the emit path because the
                # receipt body is already durable, but it must be visible to
                # verifiers and health monitors. Otherwise a post-action body
                # could exist without a tamper-evident authorization trail.
                error = {
                    "receipt_id": receipt.receipt_id,
                    "kind": receipt.kind,
                    "error_type": type(exc).__qualname__,
                    "message": str(exc)[:240],
                    "timestamp": time.time(),
                }
                with self._lock:
                    self._chain_append_errors.append(error)
                    self._chain_append_errors = self._chain_append_errors[-100:]
                try:
                    from core.runtime.errors import record_degradation

                    record_degradation(
                        "receipt_store",
                        exc,
                        severity="warning",
                        action="receipt body persisted but audit-chain append failed; verify_chain will fail",
                        receipt_required=False,
                    )
                except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as record_exc:
                    logger.debug("Receipt chain append degradation record failed: %s", record_exc)
        return receipt

    def get(self, receipt_id: str) -> AnyReceipt | None:
        with self._lock:
            cached = self._index.get(receipt_id)
            if cached is not None:
                return self._receipt_snapshot(cached)
            for kind in _RECEIPT_CLASSES:
                path = self.root / kind / f"{receipt_id}.json"
                if not path.exists():
                    continue
                try:
                    envelope = read_json_envelope(path)
                except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
                    continue
                body = envelope.get("payload") if isinstance(envelope, dict) else None
                if not isinstance(body, dict):
                    continue
                receipt = self._receipt_from_body(kind, body)
                if receipt is not None:
                    self._index[receipt.receipt_id] = receipt
                    self._prune_hot_index_locked(kind)
                    return self._receipt_snapshot(receipt)
            if not self._ledger_available:
                return None
            try:
                row = self._ledger_connection_locked().execute(
                    "SELECT kind, body_json FROM receipt_ledger WHERE receipt_id = ?",
                    (str(receipt_id),),
                ).fetchone()
            except sqlite3.Error:
                return None
            if row is None:
                return None
            try:
                body = json.loads(str(row[1]))
            except (json.JSONDecodeError, TypeError, ValueError):
                return None
            receipt = self._receipt_from_body(str(row[0]), body)
            if receipt is not None:
                self._index[receipt.receipt_id] = receipt
                self._prune_hot_index_locked(receipt.kind)
                return self._receipt_snapshot(receipt)
            return None

    def query_by_kind(self, kind: str) -> list[AnyReceipt]:
        if kind not in _RECEIPT_CLASSES:
            raise ValueError(f"unknown receipt kind '{kind}'")
        with self._lock:
            return [
                self._receipt_snapshot(receipt)
                for receipt in self._index.values()
                if receipt.kind == kind
            ]

    def query_recent(
        self,
        *,
        kinds: list[str] | None = None,
        limit: int = 20,
    ) -> list[AnyReceipt]:
        """Return the newest receipts across one or more kinds."""
        with self._lock:
            receipts = [self._receipt_snapshot(receipt) for receipt in self._index.values()]

        if kinds:
            allowed = {str(kind or "").strip() for kind in kinds if str(kind or "").strip()}
            receipts = [receipt for receipt in receipts if receipt.kind in allowed]

        receipts.sort(key=lambda receipt: float(getattr(receipt, "created_at", 0.0) or 0.0))
        if limit <= 0:
            return []
        return receipts[-limit:]

    def query_recent_persisted(self, kind: str, *, limit: int = 20) -> list[AnyReceipt]:
        """Read newest receipts of one kind across hot, ledger, and envelope storage."""

        if kind not in _RECEIPT_CLASSES:
            raise ValueError(f"unknown receipt kind '{kind}'")
        if limit <= 0:
            return []
        with self._lock:
            by_id = {
                receipt.receipt_id: receipt
                for receipt in self._index.values()
                if receipt.kind == kind
            }
            if kind in _HIGH_VOLUME_RECEIPT_KINDS and self._ledger_available:
                try:
                    rows = self._ledger_connection_locked().execute(
                        """
                        SELECT body_json FROM receipt_ledger
                        WHERE kind = ? ORDER BY created_at DESC LIMIT ?
                        """,
                        (kind, int(limit)),
                    ).fetchall()
                except sqlite3.Error:
                    rows = []
                for row in rows:
                    try:
                        body = json.loads(str(row[0]))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                    receipt = self._receipt_from_body(kind, body)
                    if receipt is not None:
                        by_id[receipt.receipt_id] = receipt

            kind_dir = self.root / kind
            if kind_dir.exists():
                files: list[tuple[float, Path]] = []
                for path in kind_dir.glob("*.json"):
                    try:
                        files.append((path.stat().st_mtime, path))
                    except OSError:
                        continue
                for _mtime, path in sorted(files, reverse=True)[: int(limit)]:
                    try:
                        envelope = read_json_envelope(path)
                    except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
                        continue
                    body = envelope.get("payload") if isinstance(envelope, dict) else None
                    if not isinstance(body, dict):
                        continue
                    receipt = self._receipt_from_body(kind, body)
                    if receipt is not None:
                        by_id[receipt.receipt_id] = receipt

            receipts = sorted(
                by_id.values(),
                key=lambda receipt: float(getattr(receipt, "created_at", 0.0) or 0.0),
            )
            return receipts[-int(limit) :]

    def reload_from_disk(self) -> int:
        """Rebuild the bounded hot index from durable receipt storage."""
        count = 0
        with self._lock:
            self._index.clear()
            for kind, cls in _RECEIPT_CLASSES.items():
                kind_dir = self.root / kind
                if kind_dir.exists():
                    files: list[tuple[float, Path]] = []
                    for path in kind_dir.glob("*.json"):
                        try:
                            files.append((path.stat().st_mtime, path))
                        except OSError:
                            continue
                    for _mtime, jf in sorted(files, reverse=True)[: self.hot_index_limit]:
                        try:
                            env = read_json_envelope(jf)
                            payload = dict(env.get("payload") or {})
                            # Strip kind from payload to avoid passing twice.
                            payload.pop("kind", None)
                            receipt = cls(**payload)
                            receipt.kind = kind
                            self._index[receipt.receipt_id] = receipt
                        except (OSError, ConnectionError, TimeoutError):
                            continue
                if kind in _HIGH_VOLUME_RECEIPT_KINDS and self._ledger_available:
                    rows = self._ledger_connection_locked().execute(
                        """
                        SELECT body_json FROM receipt_ledger
                        WHERE kind = ? ORDER BY created_at DESC LIMIT ?
                        """,
                        (kind, self.hot_index_limit),
                    ).fetchall()
                    for row in reversed(rows):
                        try:
                            body = json.loads(str(row[0]))
                        except (json.JSONDecodeError, TypeError, ValueError):
                            continue
                        receipt = self._receipt_from_body(kind, body)
                        if receipt is not None:
                            self._index[receipt.receipt_id] = receipt
                self._prune_hot_index_locked(kind)
            count = len(self._index)
        return count

    def coverage_stats(self) -> dict[str, int]:
        with self._lock:
            stats: dict[str, int] = {kind: 0 for kind in _RECEIPT_CLASSES}
            for kind in _RECEIPT_CLASSES:
                kind_dir = self.root / kind
                if kind_dir.exists():
                    stats[kind] = sum(1 for _ in kind_dir.glob("*.json"))
            if self._ledger_available:
                rows = self._ledger_connection_locked().execute(
                    "SELECT kind, COUNT(*) FROM receipt_ledger GROUP BY kind"
                ).fetchall()
                for kind, count in rows:
                    ledger_kind = str(kind)
                    stats[ledger_kind] = stats.get(ledger_kind, 0) + int(count)
            return stats

    def _persisted_receipt_kinds_locked(self) -> dict[str, str]:
        persisted: dict[str, str] = {}
        for kind in _RECEIPT_CLASSES:
            kind_dir = self.root / kind
            if not kind_dir.exists():
                continue
            for path in kind_dir.glob("*.json"):
                persisted[path.stem] = kind
        if self._ledger_available:
            try:
                rows = self._ledger_connection_locked().execute(
                    "SELECT receipt_id, kind FROM receipt_ledger"
                ).fetchall()
            except sqlite3.Error:
                rows = []
            for receipt_id, kind in rows:
                persisted[str(receipt_id)] = str(kind)
        return persisted

    def _load_body_from_disk(self, receipt_id: str, kind: str) -> dict[str, Any] | None:
        """Re-read a receipt body from disk for chain verification."""
        path = self.root / kind / f"{receipt_id}.json"
        if path.exists():
            try:
                env = read_json_envelope(path)
            except (RuntimeError, AttributeError, TypeError, ValueError):
                return None
            payload = env.get("payload") if isinstance(env, dict) else None
            if not isinstance(payload, dict):
                return None
            payload.setdefault("kind", kind)
            return payload
        with self._lock:
            return self._ledger_body_locked(receipt_id, kind)

    def storage_stats(self) -> dict[str, Any]:
        """Return hot/cold receipt storage counts for health and diagnostics."""

        with self._lock:
            hot_by_kind: dict[str, int] = {}
            for receipt in self._index.values():
                hot_by_kind[receipt.kind] = hot_by_kind.get(receipt.kind, 0) + 1
            ledger_by_kind: dict[str, int] = {}
            if self._ledger_available:
                for kind, count in self._ledger_connection_locked().execute(
                    "SELECT kind, COUNT(*) FROM receipt_ledger GROUP BY kind"
                ).fetchall():
                    ledger_by_kind[str(kind)] = int(count)
            envelope_by_kind = {
                kind: sum(1 for _ in (self.root / kind).glob("*.json"))
                for kind in _RECEIPT_CLASSES
                if (self.root / kind).exists()
            }
            return {
                "hot_index_limit": self.hot_index_limit,
                "hot_index_total": len(self._index),
                "hot_by_kind": hot_by_kind,
                "high_volume_ledger_available": self._ledger_available,
                "ledger_path": str(self._ledger_path),
                "ledger_by_kind": ledger_by_kind,
                "envelope_by_kind": envelope_by_kind,
                "persisted_total": sum(ledger_by_kind.values())
                + sum(envelope_by_kind.values()),
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._ledger is not None:
                connection = self._ledger
                self._ledger = None
                self._ledger_pid = 0
                try:
                    connection.commit()
                except sqlite3.Error as exc:
                    # Already closed, or unusable. Shutdown must not raise:
                    # the tracked-connection registry can close this handle
                    # from outside (test teardown, runtime shutdown), and a
                    # store's own close() failing on a connection someone
                    # else already released turns cleanup into an incident.
                    logger.debug("Receipt ledger commit during close failed: %s", exc)
                finally:
                    with contextlib.suppress(sqlite3.Error):
                        connection.close()
            self._ledger_available = False
            if self._chain is not None:
                chain = self._chain
                self._chain = None
                try:
                    chain.flush()
                except OSError as exc:
                    logger.warning("Could not flush receipt audit chain during close: %s", exc)
                chain.close()

    def verify_chain(self) -> dict[str, Any]:
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
        problems = list(problems)
        entries = self._chain.entries()
        chained_ids = {entry.receipt_id for entry in entries}
        with self._lock:
            persisted = self._persisted_receipt_kinds_locked()
            append_errors = list(self._chain_append_errors)
        missing_from_chain = sorted(set(persisted) - chained_ids)
        for receipt_id in missing_from_chain:
            problems.append(
                {
                    "reason": "receipt missing from audit chain",
                    "receipt_id": receipt_id,
                    "kind": persisted.get(receipt_id, "unknown"),
                }
            )
        for error in append_errors:
            problems.append({"reason": "chain append failed", **error})
        ok = bool(ok and not missing_from_chain and not append_errors)
        return {
            "ok": ok,
            "length": self._chain.length(),
            "head_hash": self._chain.head_hash(),
            "problems": problems,
            "chain_append_errors": append_errors,
            "missing_from_chain": missing_from_chain,
        }

    def export_chain(self, dest_dir: Path) -> dict[str, Any]:
        """Export the chain (chain.jsonl + MANIFEST.txt) to ``dest_dir``."""
        if self._chain is None:
            raise RuntimeError("chain not initialised")
        return cast(dict[str, Any], self._chain.export(dest_dir))


_global_store: ReceiptStore | None = None
_singleton_lock = threading.RLock()


def get_receipt_store(root: Path | None = None) -> ReceiptStore:
    global _global_store
    with _singleton_lock:
        if _global_store is None:
            _global_store = ReceiptStore(root)
        return _global_store


def reset_receipt_store() -> None:
    global _global_store
    with _singleton_lock:
        if _global_store is not None:
            _global_store.close()
        _global_store = None


def close_receipt_store() -> dict[str, object]:
    """Close the process-global store without constructing a new one."""

    with _singleton_lock:
        store = _global_store
    if store is None:
        return {"clean": True, "closed": False, "reason": "not_initialized"}
    try:
        store.close()
    except (OSError, RuntimeError, sqlite3.Error) as exc:
        return {
            "clean": False,
            "closed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"clean": True, "closed": True}
