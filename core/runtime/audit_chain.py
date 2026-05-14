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
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


GENESIS_PREV_HASH = "sha256:" + "0" * 64
SCHEMA_VERSION = 1


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON encoding suitable for hashing.
    
    [PERF] Optimized for small objects by using separators and 
    conditional sort_keys.
    """
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
        self.lock_path = self.root / ".chain.lock"
        self._lock = threading.RLock()
        self._head_hash: str = GENESIS_PREV_HASH
        self._next_seq: int = 0
        self._unsynced_entries: int = 0
        self._known_chain_signature: Optional[Tuple[int, int, int]] = None
        self._append_fd: Optional[int] = None
        self._lock_fd: Optional[int] = None
        self._root_ready = False
        self._sync_policy = os.environ.get("AURA_AUDIT_CHAIN_SYNC", "batch").strip().lower()
        try:
            self._sync_every = max(1, int(os.environ.get("AURA_AUDIT_CHAIN_FSYNC_EVERY", "32")))
        except (TypeError, ValueError):
            self._sync_every = 32
        # Best-effort load of existing chain head so a restarted process
        # extends rather than forks.  If the file is malformed, callers
        # are expected to invoke verify() and decide how to recover.
        self._load_head()

    # ------------------------------------------------------------------
    # construction-time helpers
    # ------------------------------------------------------------------
    def _load_head(self) -> None:
        with self._lock:
            self._refresh_head_from_disk_locked()

    def _refresh_head_from_disk_locked(self) -> None:
        last_record = self._read_last_record()
        if last_record is not None:
            self._head_hash = last_record["entry_hash"]
            self._next_seq = int(last_record["seq"]) + 1
        elif not self.path.exists() or self.path.stat().st_size == 0:
            self._head_hash = GENESIS_PREV_HASH
            self._next_seq = 0
        self._known_chain_signature = self._chain_signature()

    def _chain_signature(self) -> Optional[Tuple[int, int, int]]:
        """Return a cheap signature for detecting external chain writes."""
        try:
            st = self.path.stat()
            return (
                int(getattr(st, "st_ino", 0) or 0),
                int(st.st_size),
                int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))),
            )
        except OSError:
            return None

    def _refresh_head_if_disk_changed_locked(self) -> None:
        signature = self._chain_signature()
        if signature == self._known_chain_signature:
            return
        self._refresh_head_from_disk_locked()

    def _read_last_record(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            with open(self.path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                end = fh.tell()
                if end <= 0:
                    return None
                chunk_size = 4096
                buffer = b""
                pos = end
                while pos > 0:
                    read_size = min(chunk_size, pos)
                    pos -= read_size
                    fh.seek(pos)
                    buffer = fh.read(read_size) + buffer
                    lines = [line for line in buffer.splitlines() if line.strip()]
                    if len(lines) >= 2 or pos == 0:
                        return json.loads(lines[-1].decode("utf-8"))
        except (OSError, json.JSONDecodeError, IndexError, KeyError, ValueError):
            # Fall back to the slower full iterator so malformed tails still
            # surface through the existing ChainTamperError path when possible.
            last_record: Optional[Dict[str, Any]] = None
            for record in self._iter_entries():
                last_record = record
            return last_record
        return None

    @contextmanager
    def _process_append_lock(self):
        fd = self._lock_fd_for_append_locked()
        try:
            try:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass

    def _ensure_root_ready_locked(self) -> None:
        if not self._root_ready:
            self.root.mkdir(parents=True, exist_ok=True)
            self._root_ready = True

    def _lock_fd_for_append_locked(self) -> int:
        self._ensure_root_ready_locked()
        if self._lock_fd is None:
            self._lock_fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        return self._lock_fd

    def _append_fd_locked(self) -> tuple[int, bool]:
        self._ensure_root_ready_locked()
        created = not self.path.exists()
        if self._append_fd is None:
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            self._append_fd = os.open(str(self.path), flags, 0o600)
        return self._append_fd, created

    def _iter_entries(self) -> Iterator[Dict[str, Any]]:
        """Yield raw records from the chain file. Avoids dataclass overhead."""
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    raise ChainTamperError(
                        f"chain entry is not valid JSON: {line[:120]!r}"
                    )

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

        with self._lock:
            with self._process_append_lock():
                self._refresh_head_if_disk_changed_locked()
                content_hash = hash_receipt_body(body)
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
        line = json.dumps(entry.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
        # O_APPEND on POSIX guarantees the append is atomic for writes
        # under PIPE_BUF; one JSON line is well under that on every Unix.
        fd, created = self._append_fd_locked()
        os.write(fd, line.encode("utf-8"))
        self._unsynced_entries += 1
        synced_now = False
        if self._should_fsync_now():
            self._fsync_fd(fd)
            self._unsynced_entries = 0
            synced_now = True
        if created and (self._sync_policy == "always" or synced_now):
            self._fsync_dir()
        self._known_chain_signature = self._chain_signature()

    def close(self) -> None:
        for attr in ("_append_fd", "_lock_fd"):
            fd = getattr(self, attr, None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, attr, None)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _should_fsync_now(self) -> bool:
        if self._sync_policy in {"0", "false", "off", "none", "never"}:
            return False
        if self._sync_policy in {"1", "true", "on", "always", "sync"}:
            return True
        return self._unsynced_entries >= self._sync_every

    @staticmethod
    def _fsync_fd(fd: int) -> None:
        try:
            os.fsync(fd)
        except OSError:
            pass

    def _fsync_dir(self) -> None:
        # Best-effort dir fsync so newly-created chain files survive a crash.
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

    def flush(self) -> None:
        """Force pending batched audit-chain bytes to stable storage."""
        with self._lock:
            if self._unsynced_entries <= 0 or not self.path.exists():
                return
            fd = os.open(str(self.path), os.O_RDONLY)
            try:
                self._fsync_fd(fd)
                self._unsynced_entries = 0
            finally:
                os.close(fd)
            self._fsync_dir()

    def head_hash(self) -> str:
        with self._lock:
            return self._head_hash

    def length(self) -> int:
        with self._lock:
            return self._next_seq

    def entries(self) -> List[ChainEntry]:
        with self._lock:
            return [ChainEntry(**r) for r in self._iter_entries()]

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
            for record in self._iter_entries():
                r_seq = record["seq"]
                r_kind = record["kind"]
                r_id = record["receipt_id"]
                r_prev = record["prev_hash"]
                r_entry_hash = record["entry_hash"]
                r_content_hash = record["content_hash"]
                r_ts = record["timestamp"]

                if r_seq != expected_seq:
                    problems.append(
                        {
                            "seq": r_seq,
                            "kind": r_kind,
                            "receipt_id": r_id,
                            "reason": (
                                f"out-of-order or missing seq: expected "
                                f"{expected_seq}, got {r_seq}"
                            ),
                        }
                    )
                if r_prev != prev_hash:
                    problems.append(
                        {
                            "seq": r_seq,
                            "kind": r_kind,
                            "receipt_id": r_id,
                            "reason": "broken chain link (prev_hash mismatch)",
                        }
                    )
                recomputed = ChainEntry.compute_entry_hash(
                    seq=r_seq,
                    receipt_id=r_id,
                    kind=r_kind,
                    content_hash=r_content_hash,
                    timestamp=r_ts,
                    prev_hash=r_prev,
                )
                if recomputed != r_entry_hash:
                    problems.append(
                        {
                            "seq": r_seq,
                            "kind": r_kind,
                            "receipt_id": r_id,
                            "reason": "entry_hash mismatch (entry tampered)",
                        }
                    )
                if body_loader is not None:
                    body = body_loader(r_id, r_kind)
                    if body is None:
                        problems.append(
                            {
                                "seq": r_seq,
                                "kind": r_kind,
                                "receipt_id": r_id,
                                "reason": "receipt body missing on disk",
                            }
                        )
                    else:
                        actual = hash_receipt_body(body)
                        if actual != r_content_hash:
                            problems.append(
                                {
                                    "seq": r_seq,
                                    "kind": r_kind,
                                    "receipt_id": r_id,
                                    "reason": (
                                        "content_hash mismatch (receipt "
                                        "body modified)"
                                    ),
                                }
                            )
                prev_hash = r_entry_hash
                expected_seq = r_seq + 1

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
            self.flush()
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
