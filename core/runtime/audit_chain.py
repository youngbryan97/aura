"""Append-only Merkle/hash chain over receipts for tamper-evident audit.

Receipts in ``core/runtime/receipts.py`` are stored per-kind, per-id as
discrete JSON envelopes.  That format is good for queryability but does
not let an auditor detect after-the-fact tampering: an attacker with
write access to ``~/.aura/receipts/`` could edit a receipt or delete one
without anyone noticing.

This module adds a sidecar append-only chain that records, for every
receipt the ``ReceiptStore`` emits, a tamper-evident link:

    seq:           monotonically increasing per-store
    receipt_id:    receipt id this entry covers
    kind:          receipt kind (turn, governance, ...)
    content_hash:  SHA-256 of the canonical JSON of the receipt body
    timestamp:     wall-clock time when the entry was appended
    prev_hash:     entry_hash of the previous entry (genesis is all zeros)
    entry_hash:    SHA-256 over the canonical concatenation of the above

Verification walks the chain from genesis, recomputing entry hashes and
re-hashing the on-disk receipt bodies; any mismatch identifies the
tampered entry.  Deletion shows up as a seq gap; insertion shows up as
a broken link because entry_hash is over prev_hash.

The chain is sidecar (not invasive): existing receipt callers see no
change.  Only ``ReceiptStore.emit`` is extended to call ``append`` once
the underlying file is durable on disk.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


GENESIS_PREV_HASH = "sha256:" + "0" * 64
SCHEMA_VERSION = 1


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON encoding suitable for hashing."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def hash_receipt_body(body: Dict[str, Any]) -> str:
    """Hash a receipt dict (as returned by ``_ReceiptBase.to_dict``)."""
    return sha256_hex(canonical_json(body))


@dataclass
class ChainEntry:
    seq: int
    receipt_id: str
    kind: str
    content_hash: str
    timestamp: float
    prev_hash: str
    entry_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def compute_entry_hash(
        seq: int,
        receipt_id: str,
        kind: str,
        content_hash: str,
        timestamp: float,
        prev_hash: str,
    ) -> str:
        payload = {
            "seq": seq,
            "receipt_id": receipt_id,
            "kind": kind,
            "content_hash": content_hash,
            "timestamp": timestamp,
            "prev_hash": prev_hash,
        }
        return sha256_hex(canonical_json(payload))


class ChainTamperError(RuntimeError):
    """Raised when chain verification finds a corrupted or tampered entry."""

    def __init__(self, message: str, *, seq: Optional[int] = None, kind: Optional[str] = None):
        super().__init__(message)
        self.seq = seq
        self.kind = kind


class AuditChain:
    """Append-only chain stored as JSONL at ``root/_chain.jsonl``.

    The chain is its own thread-safe writer.  ``append`` is called from
    ``ReceiptStore.emit`` after the receipt body has been durably written.
    ``verify`` walks the chain and re-hashes receipt bodies to detect
    tampering.  ``export`` produces a portable bundle (chain.jsonl +
    MANIFEST.txt) suitable for offline auditing.
    """

    CHAIN_FILENAME = "_chain.jsonl"

    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / self.CHAIN_FILENAME
        self._lock = threading.RLock()
        self._head_hash: str = GENESIS_PREV_HASH
        self._next_seq: int = 0
        # Best-effort load of existing chain head so a restarted process
        # extends rather than forks.  If the file is malformed, callers
        # are expected to invoke verify() and decide how to recover.
        self._load_head()

    # ------------------------------------------------------------------
    # construction-time helpers
    # ------------------------------------------------------------------
    def _load_head(self) -> None:
        if not self.path.exists():
            return
        last_entry: Optional[ChainEntry] = None
        with self._lock:
            for entry in self._iter_entries():
                last_entry = entry
            if last_entry is not None:
                self._head_hash = last_entry.entry_hash
                self._next_seq = last_entry.seq + 1

    def _iter_entries(self) -> Iterator[ChainEntry]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    raise ChainTamperError(
                        f"chain entry is not valid JSON: {line[:120]!r}"
                    )
                yield ChainEntry(**record)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def append(
        self,
        *,
        receipt_id: str,
        kind: str,
        body: Dict[str, Any],
        timestamp: float,
    ) -> ChainEntry:
        """Append one entry to the chain.  Thread-safe and durable."""
        if not receipt_id:
            raise ValueError("receipt_id is required")
        if not kind:
            raise ValueError("kind is required")

        content_hash = hash_receipt_body(body)
        with self._lock:
            seq = self._next_seq
            prev_hash = self._head_hash
            entry_hash = ChainEntry.compute_entry_hash(
                seq=seq,
                receipt_id=receipt_id,
                kind=kind,
                content_hash=content_hash,
                timestamp=timestamp,
                prev_hash=prev_hash,
            )
            entry = ChainEntry(
                seq=seq,
                receipt_id=receipt_id,
                kind=kind,
                content_hash=content_hash,
                timestamp=timestamp,
                prev_hash=prev_hash,
                entry_hash=entry_hash,
            )
            self._durable_append(entry)
            self._head_hash = entry_hash
            self._next_seq = seq + 1
            return entry

    def _durable_append(self, entry: ChainEntry) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
        # O_APPEND on POSIX guarantees the append is atomic for writes
        # under PIPE_BUF; one JSON line is well under that on every Unix.
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        fd = os.open(str(self.path), flags, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
            try:
                os.fsync(fd)
            except OSError:
                pass
        finally:
            os.close(fd)
        # Best-effort dir fsync so the entry survives a crash.
        if hasattr(os, "O_DIRECTORY"):
            try:
                dir_fd = os.open(str(self.root), os.O_DIRECTORY)
            except OSError:
                return
            try:
                try:
                    os.fsync(dir_fd)
                except OSError:
                    pass
            finally:
                os.close(dir_fd)

    def head_hash(self) -> str:
        with self._lock:
            return self._head_hash

    def length(self) -> int:
        with self._lock:
            return self._next_seq

    def entries(self) -> List[ChainEntry]:
        with self._lock:
            return list(self._iter_entries())

    def verify(
        self,
        *,
        body_loader: Optional[callable] = None,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Walk the chain, recompute entry hashes, and (optionally)
        re-hash receipt bodies.

        ``body_loader`` is a callable ``(receipt_id, kind) -> dict | None``
        that returns the on-disk body for a receipt.  When supplied, the
        verifier compares the recomputed body hash against the chain's
        ``content_hash``; mismatches are reported.

        Returns ``(ok, problems)``.  ``problems`` is a list of dicts each
        with ``seq``, ``kind``, ``receipt_id``, ``reason``.
        """
        problems: List[Dict[str, Any]] = []
        prev_hash = GENESIS_PREV_HASH
        expected_seq = 0

        with self._lock:
            for entry in self._iter_entries():
                if entry.seq != expected_seq:
                    problems.append(
                        {
                            "seq": entry.seq,
                            "kind": entry.kind,
                            "receipt_id": entry.receipt_id,
                            "reason": (
                                f"out-of-order or missing seq: expected "
                                f"{expected_seq}, got {entry.seq}"
                            ),
                        }
                    )
                if entry.prev_hash != prev_hash:
                    problems.append(
                        {
                            "seq": entry.seq,
                            "kind": entry.kind,
                            "receipt_id": entry.receipt_id,
                            "reason": "broken chain link (prev_hash mismatch)",
                        }
                    )
                recomputed = ChainEntry.compute_entry_hash(
                    seq=entry.seq,
                    receipt_id=entry.receipt_id,
                    kind=entry.kind,
                    content_hash=entry.content_hash,
                    timestamp=entry.timestamp,
                    prev_hash=entry.prev_hash,
                )
                if recomputed != entry.entry_hash:
                    problems.append(
                        {
                            "seq": entry.seq,
                            "kind": entry.kind,
                            "receipt_id": entry.receipt_id,
                            "reason": "entry_hash mismatch (entry tampered)",
                        }
                    )
                if body_loader is not None:
                    body = body_loader(entry.receipt_id, entry.kind)
                    if body is None:
                        problems.append(
                            {
                                "seq": entry.seq,
                                "kind": entry.kind,
                                "receipt_id": entry.receipt_id,
                                "reason": "receipt body missing on disk",
                            }
                        )
                    else:
                        actual = hash_receipt_body(body)
                        if actual != entry.content_hash:
                            problems.append(
                                {
                                    "seq": entry.seq,
                                    "kind": entry.kind,
                                    "receipt_id": entry.receipt_id,
                                    "reason": (
                                        "content_hash mismatch (receipt "
                                        "body modified)"
                                    ),
                                }
                            )
                prev_hash = entry.entry_hash
                expected_seq = entry.seq + 1

        return (len(problems) == 0, problems)

    def export(self, dest_dir: Path) -> Dict[str, Any]:
        """Write a portable copy of the chain plus a MANIFEST.

        The export is suitable for handing to an external auditor: the
        chain file is byte-identical to what's in the live store, and
        the MANIFEST records head hash, length, schema version, and
        the source path.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        chain_dst = dest_dir / "chain.jsonl"
        manifest_dst = dest_dir / "MANIFEST.txt"

        with self._lock:
            head = self._head_hash
            length = self._next_seq
            if self.path.exists():
                with open(self.path, "rb") as src, open(chain_dst, "wb") as dst:
                    dst.write(src.read())
            else:
                chain_dst.write_bytes(b"")

        manifest = (
            f"audit_chain_export\n"
            f"schema_version={SCHEMA_VERSION}\n"
            f"length={length}\n"
            f"head_hash={head}\n"
            f"source_path={self.path}\n"
        )
        manifest_dst.write_text(manifest, encoding="utf-8")
        return {
            "chain_path": str(chain_dst),
            "manifest_path": str(manifest_dst),
            "length": length,
            "head_hash": head,
        }
