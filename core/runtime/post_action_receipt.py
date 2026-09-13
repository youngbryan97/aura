"""core/runtime/post_action_receipt.py — Post-Action Receipts.

Enforces proof that every governed action creates a corresponding post-action receipt.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import (
    atomic_append_text,
    durable_unlink,
    ensure_private_directory,
)
from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare

logger = logging.getLogger("Aura.PostActionReceipt")
_HOT_LIMIT_FLAG = declare(
    "AURA_POST_ACTION_RECEIPT_HOT_LIMIT",
    kind=FlagKind.INT,
    default=2048,
    description="Maximum post-action receipts retained in the process hot index",
    owner="core.runtime.post_action_receipt",
)
_MAX_BYTES_FLAG = declare(
    "AURA_POST_ACTION_RECEIPT_MAX_BYTES",
    kind=FlagKind.INT,
    default=262_144,
    description="Maximum serialized size of one post-action receipt",
    owner="core.runtime.post_action_receipt",
)
_HOT_LOAD_MAX_BYTES = 32 * 1024 * 1024
_RECEIPT_LOAD_ERRORS = (
    json.JSONDecodeError,
    TypeError,
    ValueError,
)


@dataclass
class PostActionReceipt:
    """Receipt proving the outcome and consequences of an executed action."""
    receipt_id: str
    will_receipt_id: str
    executor_name: str
    actual_outcome: str  # success / failure / partial / timeout
    output_hash: str
    error_status: str
    welfare_transaction_id: str
    body_delta: dict[str, float] = field(default_factory=dict)
    memory_delta: dict[str, Any] = field(default_factory=dict)
    rollback_target: str | None = None
    status: str = ""
    effect_verified: bool = False
    action_expectation: dict[str, Any] = field(default_factory=dict)
    verification_evidence: dict[str, Any] = field(default_factory=dict)
    action_id: str = ""
    domain: str = ""
    source: str = ""
    request_digest: str = ""
    transport_succeeded: bool = False
    retry_safe: bool = False
    manual_reconciliation_required: bool = False
    welfare_transaction_completed: bool = True
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PostActionReceiptStore:
    """Thread-safe persistent store for post-action receipts."""

    _instance: PostActionReceiptStore | None = None
    _instance_lock = threading.Lock()

    def __init__(self, persist_path: Path | None = None) -> None:
        from core.config import config
        self.persist_path = Path(
            persist_path or config.paths.data_dir / "receipts" / "post_action_receipts.jsonl"
        )
        ensure_private_directory(self.persist_path.parent)
        self._receipts: dict[str, PostActionReceipt] = {}
        self._lock = threading.RLock()
        self._load()

    @classmethod
    def get(cls) -> PostActionReceiptStore:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def hot_limit(self) -> int:
        return max(64, int(_HOT_LIMIT_FLAG.value()))

    @property
    def max_receipt_bytes(self) -> int:
        return max(16_384, int(_MAX_BYTES_FLAG.value()))

    def _prune_locked(self) -> None:
        while len(self._receipts) > self.hot_limit:
            oldest_id = next(iter(self._receipts))
            self._receipts.pop(oldest_id, None)

    def _load(self) -> None:
        if not self.persist_path.exists():
            return
        try:
            lines = _tail_lines(
                self.persist_path,
                max_lines=self.hot_limit,
                max_bytes=_HOT_LOAD_MAX_BYTES,
            )
            for tail_index, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    receipt = PostActionReceipt(**data)
                    self._receipts[receipt.receipt_id] = _receipt_snapshot(receipt)
                    self._prune_locked()
                except _RECEIPT_LOAD_ERRORS as exc:
                    record_degradation(
                        "post_action_receipt",
                        exc,
                        action=f"skipped invalid receipt ledger tail item {tail_index}",
                    )
                    logger.warning(
                        "Skipped invalid post-action receipt tail item %s: %s",
                        tail_index,
                        exc,
                    )
        except OSError as exc:
            record_degradation("post_action_receipt", exc)
            logger.warning("Failed to read post-action receipts: %s", exc)

    def record(self, receipt: PostActionReceipt) -> None:
        """Record and persist a new post-action receipt."""
        stored = _receipt_snapshot(receipt)
        _validate_receipt(stored)
        line = json.dumps(
            stored.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ) + "\n"
        encoded = line.encode("utf-8")
        if len(encoded) > self.max_receipt_bytes:
            raise ValueError(
                "post-action receipt exceeds maximum serialized size: "
                f"{len(encoded)} > {self.max_receipt_bytes}"
            )
        with self._lock:
            if stored.receipt_id in self._receipts:
                raise ValueError(f"duplicate post-action receipt_id: {stored.receipt_id}")
            try:
                atomic_append_text(self.persist_path, line)
            except OSError as exc:
                record_degradation("post_action_receipt", exc)
                logger.warning("Failed to write post-action receipt: %s", exc)
                raise
            self._receipts[stored.receipt_id] = stored
            self._prune_locked()

    async def record_async(self, receipt: PostActionReceipt) -> None:
        await asyncio.to_thread(self.record, receipt)

    def get_receipt(self, receipt_id: str) -> PostActionReceipt | None:
        with self._lock:
            receipt = self._receipts.get(receipt_id)
            return PostActionReceipt(**receipt.to_dict()) if receipt is not None else None

    def get_by_will_id(self, will_receipt_id: str) -> list[PostActionReceipt]:
        with self._lock:
            return [
                PostActionReceipt(**receipt.to_dict())
                for receipt in self._receipts.values()
                if receipt.will_receipt_id == will_receipt_id
            ]

    def list_receipts(self) -> list[PostActionReceipt]:
        with self._lock:
            return [PostActionReceipt(**receipt.to_dict()) for receipt in self._receipts.values()]

    def clear(self) -> None:
        with self._lock:
            self._receipts.clear()
            durable_unlink(self.persist_path, missing_ok=True)


def _receipt_snapshot(receipt: PostActionReceipt) -> PostActionReceipt:
    if not isinstance(receipt, PostActionReceipt):
        raise TypeError("receipt must be a PostActionReceipt")
    return PostActionReceipt(**receipt.to_dict())


def _validate_receipt(receipt: PostActionReceipt) -> None:
    required = {
        "receipt_id": receipt.receipt_id,
        "will_receipt_id": receipt.will_receipt_id,
        "executor_name": receipt.executor_name,
        "welfare_transaction_id": receipt.welfare_transaction_id,
        "output_hash": receipt.output_hash,
    }
    missing = [name for name, value in required.items() if not str(value or "").strip()]
    if missing:
        raise ValueError("post-action receipt missing required fields: " + ", ".join(missing))
    if not receipt.output_hash.startswith("sha256:"):
        raise ValueError("post-action receipt output_hash must use sha256")


def _tail_lines(path: Path, *, max_lines: int, max_bytes: int) -> list[str]:
    """Read only the bounded ledger tail needed for the hot index."""

    if max_lines <= 0 or max_bytes <= 0:
        return []
    with path.open("rb") as handle:
        handle.seek(0, io.SEEK_END)
        end = handle.tell()
        position = end
        chunks: list[bytes] = []
        newline_count = 0
        bytes_read = 0
        while position > 0 and newline_count <= max_lines and bytes_read < max_bytes:
            chunk_size = min(64 * 1024, position, max_bytes - bytes_read)
            if chunk_size <= 0:
                break
            position -= chunk_size
            handle.seek(position)
            chunk = handle.read(chunk_size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")
            bytes_read += len(chunk)
    data = b"".join(reversed(chunks))
    if position > 0:
        boundary = data.find(b"\n")
        data = data[boundary + 1 :] if boundary >= 0 else b""
    return [
        line.decode("utf-8", errors="replace")
        for line in data.splitlines()[-max_lines:]
    ]


def get_post_action_receipt_store() -> PostActionReceiptStore:
    return PostActionReceiptStore.get()


__all__ = [
    "PostActionReceipt",
    "PostActionReceiptStore",
    "get_post_action_receipt_store",
]
