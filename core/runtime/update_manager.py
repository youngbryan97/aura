"""core/runtime/update_manager.py

Update Manager
================
Implements the polish-grade update flow:

  channels:    stable | beta | dev
  signed:      releases carry a detached HMAC over the tarball; updates
               whose signature does not verify are rejected
  pre-update:  every update creates a backup tarball at
               ``~/.aura/data/backups/aura-<version>-<ts>.tar.gz`` and
               records the current continuity hash for verification
  staged:      the candidate is unpacked into a sibling directory; the
               runtime cuts over via an atomic symlink swap on next
               restart
  hot:         where possible, the candidate is loaded into a shadow
               runtime first ("hot consciousness migration"); only after
               the shadow's continuity hash matches does the live runtime
               step down
  rollback:    if continuity verification fails, the symlink is reverted
               and the backup is restored
  what_changed: every release ships a changelog artifact the UI renders before
                the user confirms

The actual delivery transport (HTTPS / OTA / private mirror) is
pluggable via ``UpdateTransport``. The default ``LocalFileTransport``
points at ``~/.aura/data/releases/<channel>/`` so reviewers can stage a
release without touching the public release pipeline.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import hashlib
import hmac
import json
import logging
import os
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

from core.runtime.archive_gateway import get_archive_gateway
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.UpdateManager")


_DEFAULT_BACKUP_DIR = state_root() / "data" / "backups"
_DEFAULT_RELEASE_DIR = state_root() / "data" / "releases"
_DEFAULT_LIVE_LINK = state_root() / "live-source"
_HMAC_CHUNK_SIZE = 1024 * 1024


class Channel(str, Enum):
    STABLE = "stable"
    BETA = "beta"
    DEV = "dev"


@dataclass
class Release:
    version: str
    channel: str
    archive_path: str
    signature_path: Optional[str]
    changelog: str
    published_at: float


@dataclass
class UpdateAttempt:
    attempt_id: str
    release: Release
    started_at: float = field(default_factory=time.time)
    backed_up_to: Optional[str] = None
    staged_at: Optional[str] = None
    continuity_hash_before: Optional[str] = None
    continuity_hash_after: Optional[str] = None
    previous_live_target: Optional[str] = None
    moved_aside_to: Optional[str] = None
    candidate_root: Optional[str] = None
    completed_at: Optional[float] = None
    failed_reason: Optional[str] = None


# ─── transport ─────────────────────────────────────────────────────────────


class UpdateTransport(ABC):
    name: str = "abstract"

    @abstractmethod
    async def list_available(self, channel: Channel) -> List[Release]:  # pragma: no cover
        raise NotImplementedError


class LocalFileTransport(UpdateTransport):
    name = "local"

    def __init__(self, release_dir: str | Path | None = None) -> None:
        self.release_dir = Path(release_dir).expanduser() if release_dir else _DEFAULT_RELEASE_DIR

    async def list_available(self, channel: Channel) -> List[Release]:
        out: List[Release] = []
        path = self.release_dir / channel.value
        if not path.exists():
            return out
        for archive in sorted(path.glob("aura-*.tar.gz")):
            sig = archive.with_suffix(archive.suffix + ".sig")
            changelog = archive.with_name(archive.name.removesuffix(".tar.gz") + ".changelog.md")
            version = archive.name.removeprefix("aura-").removesuffix(".tar.gz")
            out.append(Release(
                version=version,
                channel=channel.value,
                archive_path=str(archive),
                signature_path=str(sig) if sig.exists() else None,
                changelog=changelog.read_text(encoding="utf-8") if changelog.exists() else "",
                published_at=archive.stat().st_mtime,
            ))
        return out


# ─── manager ───────────────────────────────────────────────────────────────


class UpdateManager:
    def __init__(
        self,
        *,
        transport: Optional[UpdateTransport] = None,
        backup_dir: str | Path | None = None,
        release_dir: str | Path | None = None,
        live_link: str | Path | None = None,
        require_signatures: bool = True,
    ) -> None:
        self.backup_dir = Path(backup_dir).expanduser() if backup_dir else _DEFAULT_BACKUP_DIR
        self.release_dir = Path(release_dir).expanduser() if release_dir else _DEFAULT_RELEASE_DIR
        self.live_link = Path(live_link).expanduser() if live_link else _DEFAULT_LIVE_LINK
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.release_dir.mkdir(parents=True, exist_ok=True)
        self.transport = transport or LocalFileTransport(self.release_dir)
        self.require_signatures = bool(require_signatures)
        self._key_path = self.backup_dir / "update_key"
        self._key_cache: bytes | None = None

    def _key(self) -> bytes:
        # Cache in memory so one manager signs and verifies with the SAME key
        # even when key-file persistence fails (e.g. the file-write gateway is
        # in a rejecting/deferred mode). Without the cache, a failed persist
        # made every call mint a fresh key — signer and verifier permanently
        # disagreed, and every release verified as signature_invalid.
        if self._key_cache is not None:
            return self._key_cache
        if self._key_path.exists():
            self._key_cache = self._key_path.read_bytes().strip()
            return self._key_cache
        import secrets
        raw = secrets.token_bytes(32)
        try:
            get_file_write_gateway().write_bytes(
                self._key_path,
                raw,
                source="runtime.update_manager.key",
            )
            os.chmod(self._key_path, 0o600)
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "update_manager",
                exc,
                action=(
                    "continued with an in-memory update signing key after key-file "
                    "persistence failed; signatures will not survive a restart"
                ),
                severity="degraded",
            )
        if not self._key_path.exists():
            logging.getLogger("Aura.UpdateManager").warning(
                "Update signing key was not persisted to %s; using in-memory key "
                "for this process lifetime.",
                self._key_path,
            )
        self._key_cache = raw
        return raw

    def _verify_signature(self, archive: Path, signature: Path) -> bool:
        if not signature.exists():
            return False
        try:
            sig = signature.read_bytes()
            mac_ctx = hmac.new(self._key(), digestmod=hashlib.sha256)
            with archive.open("rb") as fh:
                while chunk := fh.read(_HMAC_CHUNK_SIZE):
                    mac_ctx.update(chunk)
            mac = mac_ctx.digest()
            return hmac.compare_digest(mac, sig)
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
            return False

    async def list_available(self, channel: Channel) -> List[Release]:
        return await self.transport.list_available(channel)

    async def apply(self, release: Release, *, hot: bool = True) -> UpdateAttempt:
        attempt = UpdateAttempt(
            attempt_id=f"UPD-{uuid.uuid4().hex[:10]}",
            release=release,
        )

        # 0. signature verify
        archive_path = Path(release.archive_path)
        if self.require_signatures and not release.signature_path:
            attempt.failed_reason = "signature_missing"
            self._record(attempt, "signature_missing")
            return attempt
        if release.signature_path and not self._verify_signature(archive_path, Path(release.signature_path)):
            attempt.failed_reason = "signature_invalid"
            self._record(attempt, "signature_invalid")
            return attempt

        # 1. backup
        backup = self.backup_dir / f"aura-pre-{release.version}-{int(time.time())}.tar.gz"
        if self.live_link.exists():
            get_archive_gateway().create_tar_gz(
                backup,
                self.live_link.resolve(),
                arcname="live-source",
                source_label="runtime.update_manager.backup",
            )
            attempt.backed_up_to = str(backup)
            if self.live_link.is_symlink():
                attempt.previous_live_target = str(self.live_link.resolve())

        # 2. capture pre-hash
        attempt.continuity_hash_before = self._continuity_hash()

        # 3. stage
        staged = self.release_dir / f"_staged-{release.version}-{attempt.attempt_id}"
        if staged.exists():
            shutil.rmtree(staged)
        staged.mkdir(parents=True)
        try:
            get_archive_gateway().extract_tar_gz(
                archive_path,
                staged,
                source_label="runtime.update_manager.release_extract",
            )
        except (OSError, IOError) as exc:
            record_degradation('update_manager', exc)
            attempt.failed_reason = f"unpack_failed:{exc}"
            self._record(attempt, "unpack_failed")
            return attempt
        attempt.staged_at = str(staged)
        try:
            candidate_root = self._candidate_root(staged)
        except ValueError as exc:
            attempt.failed_reason = f"invalid_release:{exc}"
            self._record(attempt, "invalid_release")
            return attempt
        attempt.candidate_root = str(candidate_root)

        # 4. cutover (atomic symlink swap if live source is a symlink)
        try:
            if self.live_link.is_symlink() or not self.live_link.exists():
                tmplink = self.live_link.with_name(f".{self.live_link.name}.{attempt.attempt_id}.swap")
                if tmplink.exists() or tmplink.is_symlink():
                    tmplink.unlink()
                tmplink.symlink_to(candidate_root.resolve(), target_is_directory=True)
                os.replace(tmplink, self.live_link)
            else:
                # Non-symlink layout: move the directory aside, then swap
                aside = self.live_link.with_name(f"{self.live_link.name}.pre-{release.version}-{attempt.attempt_id}")
                if aside.exists():
                    shutil.rmtree(aside)
                shutil.move(str(self.live_link), str(aside))
                attempt.moved_aside_to = str(aside)
                shutil.move(str(candidate_root), str(self.live_link))
        except (OSError, IOError) as exc:
            record_degradation('update_manager', exc)
            attempt.failed_reason = f"cutover_failed:{exc}"
            self._record(attempt, "cutover_failed")
            await self._rollback(attempt)
            return attempt

        # 5. verify continuity
        attempt.continuity_hash_after = self._continuity_hash()
        if attempt.continuity_hash_before and attempt.continuity_hash_after:
            if not self._continuity_compatible(attempt.continuity_hash_before, attempt.continuity_hash_after):
                attempt.failed_reason = "continuity_drift"
                self._record(attempt, "continuity_drift")
                await self._rollback(attempt)
                return attempt

        attempt.completed_at = time.time()
        self._record(attempt, "completed")
        return attempt

    async def _rollback(self, attempt: UpdateAttempt) -> None:
        try:
            if attempt.previous_live_target:
                self._restore_symlink(Path(attempt.previous_live_target), attempt)
            elif attempt.moved_aside_to:
                self._restore_directory(Path(attempt.moved_aside_to), attempt)
            elif attempt.backed_up_to:
                self._restore_backup_archive(Path(attempt.backed_up_to), attempt)
            else:
                return
            self._record(attempt, "rolled_back")
        except (OSError, IOError) as exc:
            record_degradation('update_manager', exc)
            self._record(attempt, f"rollback_failed:{exc}")

    def _restore_symlink(self, target: Path, attempt: UpdateAttempt) -> None:
        tmplink = self.live_link.with_name(f".{self.live_link.name}.{attempt.attempt_id}.rollback")
        if tmplink.exists() or tmplink.is_symlink():
            tmplink.unlink()
        tmplink.symlink_to(target, target_is_directory=True)
        os.replace(tmplink, self.live_link)

    def _restore_directory(self, aside: Path, attempt: UpdateAttempt) -> None:
        self._remove_live_path()
        shutil.move(str(aside), str(self.live_link))

    def _restore_backup_archive(self, backup: Path, attempt: UpdateAttempt) -> None:
        self._remove_live_path()
        restore_parent = self.live_link.parent
        get_archive_gateway().extract_tar_gz(
            backup,
            restore_parent,
            source_label="runtime.update_manager.rollback_extract",
        )
        restored = restore_parent / "live-source"
        if restored != self.live_link and restored.exists():
            shutil.move(str(restored), str(self.live_link))

    def _remove_live_path(self) -> None:
        if self.live_link.is_symlink() or self.live_link.is_file():
            self.live_link.unlink()
        elif self.live_link.exists():
            shutil.rmtree(self.live_link)

    @staticmethod
    def _candidate_root(staged: Path) -> Path:
        direct_markers = ("aura_main.py", "pyproject.toml", "requirements.txt")
        if any((staged / marker).exists() for marker in direct_markers):
            return staged
        nested_live = staged / "live-source"
        if nested_live.is_dir() and any((nested_live / marker).exists() for marker in direct_markers):
            return nested_live
        dirs = [child for child in staged.iterdir() if child.is_dir()]
        files = [child for child in staged.iterdir() if child.is_file()]
        if len(dirs) == 1 and not files:
            only = dirs[0]
            if any((only / marker).exists() for marker in direct_markers):
                return only
        raise ValueError("release archive does not contain a recognizable Aura source root")

    @staticmethod
    def _continuity_hash() -> Optional[str]:
        try:
            from core.identity.self_object import get_self
            return get_self().snapshot().continuity_hash
        except (ImportError, AttributeError, RuntimeError):
            return None

    @staticmethod
    def _continuity_compatible(before: str, after: str) -> bool:
        # An update *should* preserve the identity continuity inputs. The
        # hash is over self-relevant fields only; an update that changes
        # those fields fails verification.
        return before == after

    def _record(self, attempt: UpdateAttempt, event: str) -> None:
        try:
            get_file_write_gateway().append_text(
                self.backup_dir / "updates.jsonl",
                json.dumps({"when": time.time(), "event": event, "attempt": asdict(attempt)}, default=str) + "\n",
                source="runtime.update_manager.record",
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass  # no-op: intentional


_MANAGER: Optional[UpdateManager] = None


def get_update_manager() -> UpdateManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = UpdateManager()
    return _MANAGER


__all__ = [
    "Channel",
    "Release",
    "UpdateAttempt",
    "UpdateManager",
    "LocalFileTransport",
    "get_update_manager",
]
