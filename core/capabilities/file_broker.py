"""core/capabilities/file_broker.py — Sandboxed File Operations
================================================================
All file operations go through this broker.

Enforces allowlist: ~/Documents/Aura, ~/Desktop/Aura, ~/Downloads,
system temporary Aura paths, ~/.aura/data. Handles special characters, versioning,
rollback, and produces receipts for every operation.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.governance_context import local_internal_governed_scope
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.FileBroker")

# Resource bounds so a read/hash/list cannot load an arbitrarily large file or
# directory into memory.
_MAX_READ_BYTES = 32 * 1024 * 1024      # 32 MB text/binary read + hash ceiling
_MAX_LIST_ENTRIES = 5000

# Filesystem errors raised by pathlib/shutil AND the governed gateway
# (RuntimeError / policy exceptions) — the write methods previously caught only
# OSError and let governance failures escape the result contract.
_FS_ERRORS = (OSError, shutil.Error, RuntimeError, ValueError)


@dataclass
class FileOperation:
    """Record of a file operation for rollback."""
    operation: str       # "create", "move", "copy", "delete", "write"
    source: str
    destination: str = ""
    backup_path: str = ""  # for rollback
    timestamp: float = field(default_factory=time.time)


class SandboxedFileBroker:
    """All file operations go through this broker.

    Features:
    - Allowlist enforcement (only approved directories)
    - Path sanitization (special chars, traversal prevention)
    - Rollback support (undo last N operations)
    - File versioning (auto-version if file exists)
    - Receipt generation for audit
    """

    # Allowed root directories (expanded at runtime)
    ALLOWED_ROOTS = [
        "~/Documents/Aura",
        "~/Desktop/Aura",
        "~/Downloads",
        "~/.aura/data",
    ]
    TEMP_PREFIX = "aura_"

    def __init__(self) -> None:
        self._operations: list[FileOperation] = []
        self._max_operations = 200
        self._expanded_roots: list[Path] = []
        self._started = False
        # Serializes version selection + the rollback journal against
        # concurrent writers/rollbacks.
        self._lock = threading.RLock()

    async def start(self) -> None:
        if self._started:
            return
        self._expanded_roots = [
            Path(os.path.expanduser(r)) for r in self.ALLOWED_ROOTS
        ]
        ServiceContainer.register_instance("file_broker", self, required=False)
        self._started = True
        logger.info(
            "SandboxedFileBroker ONLINE — allowed roots: %s",
            [str(r) for r in self._expanded_roots],
        )

    def _is_allowed(self, path: Path) -> bool:
        """Check if a path is within an allowed root."""
        return self._resolve_allowed(path) is not None

    def _resolve_allowed(self, path: Path) -> Path | None:
        """Resolve a path and return it ONLY if it is within an allowed root.

        Returning the RESOLVED path (and using it for the subsequent operation)
        closes the check-vs-use gap: authorization and mutation act on the same
        canonicalized path instead of re-deriving an unresolved one.
        """
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError, ValueError):
            return None
        roots = self._expanded_roots or [Path(os.path.expanduser(r)) for r in self.ALLOWED_ROOTS]
        for root in roots:
            try:
                resolved.relative_to(root.resolve())
                return resolved
            except (ValueError, OSError):
                continue
        try:
            relative_to_temp = resolved.relative_to(Path(tempfile.gettempdir()).resolve())
            if relative_to_temp.parts and relative_to_temp.parts[0].startswith(self.TEMP_PREFIX):
                return resolved
        except ValueError:
            return None
        return None

    @staticmethod
    def sanitize_name(name: str) -> str:
        """Sanitize a filename for safe filesystem use.

        Handles apostrophes, special chars, and length limits.
        """
        # Replace problematic chars but keep common ones
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
        # Keep apostrophes but escape them for shell safety
        sanitized = sanitized.strip(". ")
        # Truncate to reasonable length
        if len(sanitized) > 200:
            sanitized = sanitized[:200]
        return sanitized or "unnamed"

    def _record_op(self, op: FileOperation) -> None:
        with self._lock:
            self._operations.append(op)
            if len(self._operations) > self._max_operations:
                self._operations = self._operations[-self._max_operations:]

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def create_folder(self, path: str) -> dict[str, Any]:
        """Create a folder, including parent directories."""
        p = self._resolve_allowed(Path(path).expanduser())
        if p is None:
            return {"success": False, "error": f"Path not in allowlist: {path}"}

        try:
            already_existed = await asyncio.to_thread(p.exists)
            with local_internal_governed_scope("capabilities.file_broker.create_folder", domain="file_write"):
                await asyncio.to_thread(lambda: p.mkdir(parents=True, exist_ok=True))
            # Only record ownership (for rollback) when THIS call created the
            # directory — a pre-existing folder must not be deletable by a later
            # rollback of an operation that did not create it.
            if not already_existed:
                self._record_op(FileOperation("create", str(p)))
            return {"success": True, "path": str(p), "created": not already_existed}
        except _FS_ERRORS as e:
            return {"success": False, "error": str(e)}

    def _backup_for_overwrite(self, p: Path) -> str:
        """Snapshot an existing file before overwrite so rollback can restore
        it (an in-place overwrite previously lost the original irretrievably)."""
        try:
            if not p.exists():
                return ""
            fd, tmp = tempfile.mkstemp(prefix="aura_fbbak_", suffix=p.suffix)
            os.close(fd)
            shutil.copy2(str(p), tmp)
            return tmp
        except (OSError, shutil.Error):
            return ""

    async def write_file(self, path: str, content: str, overwrite: bool = False) -> dict[str, Any]:
        """Write content to a file atomically."""
        p = self._resolve_allowed(Path(path).expanduser())
        if p is None:
            return {"success": False, "error": f"Path not in allowlist: {path}"}

        encoded = content.encode("utf-8")
        if len(encoded) > _MAX_READ_BYTES:
            return {"success": False, "error": f"Content exceeds write limit ({len(encoded)} bytes)"}
        backup_path = ""
        with self._lock:
            if p.exists() and not overwrite:
                p = self._version_path(p)
            elif overwrite:
                backup_path = await asyncio.to_thread(self._backup_for_overwrite, p)

        try:
            await get_file_write_gateway().write_text_async(
                p, content, encoding="utf-8", source="file_broker.write_file",
            )
            file_hash = hashlib.sha256(encoded).hexdigest()
            self._record_op(FileOperation("write", str(p), backup_path=backup_path))
            return {"success": True, "path": str(p), "size": len(encoded), "hash": file_hash}
        except _FS_ERRORS as e:
            return {"success": False, "error": str(e)}

    async def write_bytes(self, path: str, data: bytes, overwrite: bool = False) -> dict[str, Any]:
        """Write binary data to a file."""
        p = self._resolve_allowed(Path(path).expanduser())
        if p is None:
            return {"success": False, "error": f"Path not in allowlist: {path}"}
        if len(data) > _MAX_READ_BYTES:
            return {"success": False, "error": f"Content exceeds write limit ({len(data)} bytes)"}

        backup_path = ""
        with self._lock:
            if p.exists() and not overwrite:
                p = self._version_path(p)
            elif overwrite:
                backup_path = await asyncio.to_thread(self._backup_for_overwrite, p)

        try:
            await get_file_write_gateway().write_bytes_async(
                p, data, source="file_broker.write_bytes",
            )
            file_hash = hashlib.sha256(data).hexdigest()
            self._record_op(FileOperation("write", str(p), backup_path=backup_path))
            return {"success": True, "path": str(p), "size": len(data), "hash": file_hash}
        except _FS_ERRORS as e:
            return {"success": False, "error": str(e)}

    async def move_file(self, source: str, destination: str) -> dict[str, Any]:
        """Move a file, with rollback support. BOTH ends must be allowlisted."""
        # The SOURCE must be allowlisted too — validating only the destination
        # let move_file remove arbitrary host files (and copy_file exfiltrate
        # them into an allowed location).
        src = self._resolve_allowed(Path(source).expanduser())
        if src is None:
            return {"success": False, "error": f"Source not in allowlist: {source}"}
        dst = self._resolve_allowed(Path(destination).expanduser())
        if dst is None:
            return {"success": False, "error": f"Destination not in allowlist: {destination}"}
        if not await asyncio.to_thread(src.exists):
            return {"success": False, "error": f"Source not found: {source}"}

        try:
            with self._lock:
                if dst.exists():
                    dst = self._version_path(dst)
            with local_internal_governed_scope("capabilities.file_broker.move", domain="file_write"):
                await asyncio.to_thread(lambda: dst.parent.mkdir(parents=True, exist_ok=True))
                await asyncio.to_thread(shutil.move, str(src), str(dst))
            self._record_op(FileOperation("move", str(src), str(dst)))
            return {"success": True, "source": str(src), "destination": str(dst)}
        except _FS_ERRORS as e:
            return {"success": False, "error": str(e)}

    async def copy_file(self, source: str, destination: str) -> dict[str, Any]:
        """Copy a file. BOTH ends must be allowlisted."""
        src = self._resolve_allowed(Path(source).expanduser())
        if src is None:
            return {"success": False, "error": f"Source not in allowlist: {source}"}
        dst = self._resolve_allowed(Path(destination).expanduser())
        if dst is None:
            return {"success": False, "error": f"Destination not in allowlist: {destination}"}
        if not await asyncio.to_thread(src.exists):
            return {"success": False, "error": f"Source not found: {source}"}

        try:
            with self._lock:
                if dst.exists():
                    dst = self._version_path(dst)
            with local_internal_governed_scope("capabilities.file_broker.copy", domain="file_write"):
                await asyncio.to_thread(lambda: dst.parent.mkdir(parents=True, exist_ok=True))
                await asyncio.to_thread(shutil.copy2, str(src), str(dst))
            self._record_op(FileOperation("copy", str(src), str(dst)))
            return {"success": True, "source": str(src), "destination": str(dst)}
        except _FS_ERRORS as e:
            return {"success": False, "error": str(e)}

    async def read_file(self, path: str) -> dict[str, Any]:
        """Read a text file — allowlist-enforced and size-bounded."""
        p = self._resolve_allowed(Path(path).expanduser())
        if p is None:
            return {"success": False, "error": f"Path not in allowlist: {path}"}

        def _read() -> dict[str, Any]:
            if not p.exists():
                return {"success": False, "error": f"File not found: {path}"}
            size = p.stat().st_size
            if size > _MAX_READ_BYTES:
                return {"success": False, "error": f"File exceeds read limit ({size} > {_MAX_READ_BYTES} bytes)"}
            data = p.read_bytes()[:_MAX_READ_BYTES]
            content = data.decode("utf-8", errors="replace")
            return {"success": True, "content": content, "size": len(data)}

        try:
            return await asyncio.to_thread(_read)
        except _FS_ERRORS as e:
            return {"success": False, "error": str(e)}

    async def file_exists(self, path: str) -> bool:
        p = self._resolve_allowed(Path(path).expanduser())
        if p is None:
            return False
        return await asyncio.to_thread(p.exists)

    async def file_hash(self, path: str) -> str:
        """Compute the full SHA256 hash of an allowlisted file."""
        p = self._resolve_allowed(Path(path).expanduser())
        if p is None:
            return ""

        def _hash() -> str:
            if not p.exists() or p.stat().st_size > _MAX_READ_BYTES:
                return ""
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                while chunk := fh.read(1024 * 1024):
                    h.update(chunk)
            return h.hexdigest()

        try:
            return await asyncio.to_thread(_hash)
        except _FS_ERRORS:
            return ""

    async def list_dir(self, path: str) -> dict[str, Any]:
        """List directory contents — allowlist-enforced and entry-bounded."""
        p = self._resolve_allowed(Path(path).expanduser())
        if p is None:
            return {"success": False, "error": f"Path not in allowlist: {path}"}

        def _list() -> dict[str, Any]:
            if not p.exists():
                return {"success": False, "error": f"Directory not found: {path}"}
            if not p.is_dir():
                return {"success": False, "error": f"Not a directory: {path}"}
            entries = []
            truncated = False
            for entry in sorted(p.iterdir()):
                if len(entries) >= _MAX_LIST_ENTRIES:
                    truncated = True
                    break
                try:
                    is_dir = entry.is_dir()
                    entries.append({
                        "name": entry.name,
                        "is_dir": is_dir,
                        "size": entry.stat().st_size if entry.is_file() else 0,
                    })
                except OSError:
                    continue
            return {"success": True, "entries": entries, "count": len(entries), "truncated": truncated}

        try:
            return await asyncio.to_thread(_list)
        except _FS_ERRORS as e:
            return {"success": False, "error": str(e)}

    async def reveal_in_finder(self, path: str) -> bool:
        """Open Finder and highlight the file — allowlist-enforced."""
        p = self._resolve_allowed(Path(path).expanduser())
        if p is None:
            logger.warning("reveal_in_finder refused non-allowlisted path: %s", path)
            return False
        proc = None
        try:
            proc = await get_subprocess_gateway().spawn_async(
                ["open", "-R", str(p)],
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                source="file_broker.reveal_in_finder",
                accelerator_capability="none",
            )
            await asyncio.wait_for(proc.wait(), timeout=5.0)
            return proc.returncode == 0
        except (TimeoutError, OSError):
            # Terminate + reap the child on timeout so it is not orphaned.
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except (TimeoutError, OSError, ProcessLookupError):
                    pass
            return False

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    async def rollback_last(self) -> dict[str, Any]:
        """Undo the last file operation."""
        with self._lock:
            if not self._operations:
                return {"success": False, "error": "No operations to rollback"}
            op = self._operations[-1]

        try:
            result = await asyncio.to_thread(self._apply_rollback, op)
        except _FS_ERRORS as e:
            result = {"success": False, "error": str(e)}

        # Only consume the journal entry when the inverse actually succeeded —
        # a failed inverse must not silently discard the record.
        if result.get("success"):
            with self._lock:
                if self._operations and self._operations[-1] is op:
                    self._operations.pop()
        return result

    def _apply_rollback(self, op: FileOperation) -> dict[str, Any]:
        if op.operation == "move":
            if Path(op.destination).exists() and not Path(op.source).exists():
                shutil.move(op.destination, op.source)
                return {"success": True, "rolled_back": f"move {op.destination} → {op.source}"}
            return {"success": False, "error": "move rollback preconditions not met"}
        if op.operation == "copy":
            if Path(op.destination).exists():
                os.remove(op.destination)
                return {"success": True, "rolled_back": f"removed copy {op.destination}"}
            return {"success": False, "error": "copy destination missing"}
        if op.operation == "create":
            if Path(op.source).exists() and Path(op.source).is_dir():
                try:
                    Path(op.source).rmdir()  # only removes if empty
                    return {"success": True, "rolled_back": f"removed folder {op.source}"}
                except OSError:
                    return {"success": False, "error": "Folder not empty"}
            return {"success": False, "error": "folder missing"}
        if op.operation == "write":
            target = Path(op.source)
            # If this write overwrote an existing file, restore the original
            # from the backup instead of deleting (which lost the prior content).
            if op.backup_path and Path(op.backup_path).exists():
                shutil.copy2(op.backup_path, str(target))
                Path(op.backup_path).unlink(missing_ok=True)
                return {"success": True, "rolled_back": f"restored original {target}"}
            if target.exists():
                os.remove(target)
                return {"success": True, "rolled_back": f"removed {target}"}
            return {"success": False, "error": "write target missing"}
        return {"success": False, "error": f"Cannot rollback {op.operation}"}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _version_path(self, path: Path) -> Path:
        """Add version suffix to avoid overwriting."""
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        for v in range(2, 100):
            candidate = parent / f"{stem}_v{v}{suffix}"
            if not candidate.exists():
                return candidate
        return parent / f"{stem}_{int(time.time())}{suffix}"

    def get_status(self) -> dict[str, Any]:
        return {
            "operations": len(self._operations),
            "allowed_roots": [str(r) for r in self._expanded_roots],
        }


_instance: SandboxedFileBroker | None = None


def get_file_broker() -> SandboxedFileBroker:
    global _instance
    if _instance is None:
        _instance = SandboxedFileBroker()
    return _instance


__all__ = ["SandboxedFileBroker", "get_file_broker"]
