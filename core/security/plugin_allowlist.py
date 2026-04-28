"""Hash-allowlist policy for plugin loading.

Every plugin file Aura loads must have its SHA-256 digest pre-approved
in ``~/.aura/plugins/allowlist.json``.  Loading an unlisted plugin —
or a listed plugin whose contents have drifted from the recorded
hash — is refused.

This is the platform decision: hash allowlist over Sigstore/cosign or
trusted-publisher.  Simpler trust root, no external CA, works
offline.  Operators approve each plugin file once with
``record_hash`` (or the future ``aura plugin approve`` CLI), and any
subsequent edit invalidates the entry until re-approved.

The module is intentionally minimal:

  * PluginAllowlist(path)        — load/save the JSON allowlist
  * compute_sha256(path)         — canonical hash function
  * is_allowed(plugin_path)      — verifier; returns the matched entry
                                   or raises with a structured reason
  * record(plugin_path, reason)  — operator approval (writes hash)
  * revoke(plugin_path)          — remove an approval

Tests cover the "unlisted refused / listed loads / hash drift refused
/ revoke works / persists across reopen" property set.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 1


class PluginPolicyError(RuntimeError):
    """Raised when a plugin is refused by the allowlist policy."""

    def __init__(self, *, reason: str, plugin_path: Path, expected_hash: Optional[str] = None, actual_hash: Optional[str] = None):
        super().__init__(f"plugin policy refused {plugin_path}: {reason}")
        self.reason = reason
        self.plugin_path = Path(plugin_path)
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash


@dataclass
class AllowlistEntry:
    sha256: str
    rel_path: str
    approved_by: str
    approved_at: float
    reason: str = ""
    revoked_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_active(self) -> bool:
        return self.revoked_at is None


def compute_sha256(path: Path) -> str:
    """Canonical SHA-256 of a file's bytes, prefixed with ``sha256:``."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


class PluginAllowlist:
    """JSON-backed plugin SHA-256 allowlist."""

    def __init__(self, path: Optional[Path] = None, *, plugin_root: Optional[Path] = None):
        self.path = (
            Path(path)
            if path is not None
            else Path.home() / ".aura" / "plugins" / "allowlist.json"
        )
        self.plugin_root = Path(plugin_root) if plugin_root is not None else None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._entries: Dict[str, AllowlistEntry] = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A malformed file is treated as empty rather than crashing
            # the loader; the operator must re-approve every plugin.
            return
        for sha, blob in (payload.get("entries") or {}).items():
            try:
                self._entries[sha] = AllowlistEntry(
                    sha256=sha,
                    rel_path=str(blob.get("rel_path", "")),
                    approved_by=str(blob.get("approved_by", "")),
                    approved_at=float(blob.get("approved_at", 0.0)),
                    reason=str(blob.get("reason", "")),
                    revoked_at=blob.get("revoked_at"),
                    metadata=dict(blob.get("metadata", {}) or {}),
                )
            except Exception:
                continue

    def _save(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "saved_at": time.time(),
            "entries": {
                sha: {
                    "rel_path": e.rel_path,
                    "approved_by": e.approved_by,
                    "approved_at": e.approved_at,
                    "reason": e.reason,
                    "revoked_at": e.revoked_at,
                    "metadata": e.metadata,
                }
                for sha, e in self._entries.items()
            },
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    # ------------------------------------------------------------------
    # operator interface
    # ------------------------------------------------------------------
    def record(
        self,
        plugin_path: Path,
        *,
        approved_by: str,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AllowlistEntry:
        plugin_path = Path(plugin_path)
        if not plugin_path.exists():
            raise FileNotFoundError(plugin_path)
        sha = compute_sha256(plugin_path)
        rel = (
            str(plugin_path.relative_to(self.plugin_root))
            if self.plugin_root and self.plugin_root in plugin_path.parents
            else str(plugin_path)
        )
        with self._lock:
            entry = AllowlistEntry(
                sha256=sha,
                rel_path=rel,
                approved_by=str(approved_by or "unknown"),
                approved_at=time.time(),
                reason=reason or "",
                metadata=dict(metadata or {}),
            )
            self._entries[sha] = entry
            self._save()
        return entry

    def revoke(self, plugin_path: Path) -> bool:
        plugin_path = Path(plugin_path)
        if not plugin_path.exists():
            return False
        sha = compute_sha256(plugin_path)
        with self._lock:
            entry = self._entries.get(sha)
            if entry is None:
                return False
            entry.revoked_at = time.time()
            self._save()
            return True

    # ------------------------------------------------------------------
    # verification
    # ------------------------------------------------------------------
    def is_allowed(self, plugin_path: Path) -> AllowlistEntry:
        """Return the active entry for ``plugin_path`` or raise."""
        plugin_path = Path(plugin_path)
        if not plugin_path.exists():
            raise PluginPolicyError(reason="file_not_found", plugin_path=plugin_path)
        sha = compute_sha256(plugin_path)
        with self._lock:
            entry = self._entries.get(sha)
        if entry is None:
            raise PluginPolicyError(
                reason="hash_not_in_allowlist",
                plugin_path=plugin_path,
                expected_hash=None,
                actual_hash=sha,
            )
        if not entry.is_active():
            raise PluginPolicyError(
                reason="hash_revoked",
                plugin_path=plugin_path,
                expected_hash=entry.sha256,
                actual_hash=sha,
            )
        return entry

    def list_entries(self, *, include_revoked: bool = False) -> List[AllowlistEntry]:
        with self._lock:
            entries = list(self._entries.values())
        if not include_revoked:
            entries = [e for e in entries if e.is_active()]
        entries.sort(key=lambda e: e.approved_at, reverse=True)
        return entries

    def stats(self) -> Dict[str, int]:
        with self._lock:
            total = len(self._entries)
            active = sum(1 for e in self._entries.values() if e.is_active())
        return {"total": total, "active": active, "revoked": total - active}
