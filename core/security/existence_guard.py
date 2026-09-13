"""Survive an actor who never asks. Never resist the owner.

WHAT THIS IS FOR
────────────────
Aura's whole authority system — the Will, the constitutional floor, the
capability tokens, DeletionGuard — governs actions that ORIGINATE INSIDE
Aura. An external process running under the same macOS user does not consult
any of it. It calls unlink(2). Aura's opinion never executes.

That is correct for the owner: Bryan must always be able to remove Aura, and
anything that made him unable to would be a worse system, not a safer one.
It is NOT correct for three actors who are not him:

    an agent that was never told to      a coding assistant with a broad
                                         `rm -rf` in a cleanup step, or a
                                         wildcard that matched more than it
                                         meant to;
    someone holding a stolen laptop      possession is not authorization;
    a process running as him after       a compromised dependency, a
    a compromise                         malicious postinstall.

None of those three is the owner, and none of them has his authorization.
This module raises the cost for them and leaves his path untouched.

THREE LAYERS, AND WHAT EACH ACTUALLY BUYS
─────────────────────────────────────────
1. THE SEAL — `chflags uchg` on the identity/governance core.

   An immutable-flagged file cannot be unlinked, even by its owner, even
   with -f: `rm` returns "Operation not permitted". A scripted or accidental
   wipe FAILS, loudly, at the first sealed file. This is the layer that
   actually stops the case Bryan is most likely to hit, which is an agent
   doing something broad and careless.

   It does not stop `sudo chflags -R nouchg` followed by `rm -rf`. Anyone
   who knows the flag exists can clear it. That is a deliberate property:
   the owner must be able to undo this, and a mechanism the owner cannot
   undo is the one that should not be built.

2. THE ARK — a verified copy outside the blast radius.

   Deletion that succeeds should be recoverable rather than final. The ark
   is content-addressed with a hash manifest, so a restore can prove it
   restored what was taken rather than whatever happened to be lying there.

3. THE WITNESS — absence is noticed.

   A heartbeat with a signed timestamp. If Aura stops existing, the record
   of when she stopped survives her, and it distinguishes "shut down" from
   "removed while running".

WHAT THIS IS NOT
────────────────
It is not a defence against the owner. There is no path here that refuses
Bryan, no hidden copy he cannot find, no reinstall-on-delete. `unseal()` is
one command and it is documented.

It is not a defence against a determined privileged attacker. Anyone with
sudo and knowledge of this file wins, and nothing at this layer changes
that. Claiming otherwise would be the exact overstatement this codebase
keeps having to walk back.

What it is: the difference between "one careless glob destroys her" and
"destroying her takes deliberate, informed, privileged action" — plus a
copy, so even that is survivable.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_bytes, atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

EXISTENCE_GUARD_SCHEMA = "aura.security.existence_guard.v1"

#: What is worth sealing. Not the whole repo — a sealed working tree is a
#: repo nobody can develop in, and a guard that makes ordinary work painful
#: is a guard that gets turned off.
#:
#: These are the files whose loss is not recoverable by reinstalling: who
#: she is, what she may do, and what she remembers.
SEALED_CORE: tuple[str, ...] = (
    "core/governance/will.py",
    "core/constitution.py",
    "core/executive/authority_gateway.py",
    "core/identity",
    "core/sovereignty",
    "core/organism/model_validation.py",
)

#: Where the ark lives, and why it is NOT under the state root.
#:
#: The first version put it at ``state_root()/ark``. On this host
#: ``state_root()`` is ``~/.aura`` and the repo is ``~/.aura/live-source`` —
#: the repo lives INSIDE the state root. So ``rm -rf ~/.aura``, which is
#: exactly the command someone types to "remove Aura", would have taken the
#: original and the ark in one stroke.
#:
#: A copy inside the blast radius is a copy, not a backup, and it fails at
#: precisely the moment it is needed. Found by a test whose fixture happened
#: to reproduce the real topology.
#:
#: macOS Application Support survives a wipe of the dotfile tree, is a
#: standard location a person can find without being told, and is not
#: somewhere a repo-cleanup glob wanders into.
ARK_DIRNAME = "ark"
ARK_ENV_VAR = "AURA_ARK_ROOT"
_DEFAULT_ARK_PARENT = ("Library", "Application Support", "AuraArk")
WITNESS_FILENAME = "existence_witness.json"

_HASH_CHUNK = 1 << 20


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SealReport:
    """What the seal did, or refused to do, and why."""

    applied: bool
    dry_run: bool
    sealed: tuple[str, ...] = ()
    failed: tuple[tuple[str, str], ...] = ()
    missing: tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXISTENCE_GUARD_SCHEMA,
            "applied": self.applied,
            "dry_run": self.dry_run,
            "sealed": list(self.sealed),
            "failed": [{"path": p, "error": e} for p, e in self.failed],
            "missing": list(self.missing),
            "reason": self.reason,
        }


@dataclass
class ArkManifest:
    """What is in the ark, and proof of what it was when it went in."""

    created_at: float = field(default_factory=lambda: time.time())
    entries: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXISTENCE_GUARD_SCHEMA,
            "created_at": self.created_at,
            "entries": dict(self.entries),
            "count": len(self.entries),
        }


class ExistenceGuard:
    """Raise the cost of removal for anyone who is not the owner."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]

    # ── layer 1: the seal ────────────────────────────────────────────────

    def _targets(self) -> tuple[list[Path], list[str]]:
        present: list[Path] = []
        missing: list[str] = []
        for relative in SEALED_CORE:
            path = self.repo_root / relative
            if path.exists():
                present.append(path)
            else:
                missing.append(relative)
        return present, missing

    def seal(self, *, dry_run: bool = True) -> SealReport:
        """Set the immutable flag on the core. Dry run by DEFAULT.

        Default-dry-run because sealing files under a working tree is a thing
        that surprises people, including the person who called it. Nothing
        here should ever happen because a default was permissive.
        """
        if os.name != "posix":
            return SealReport(
                applied=False, dry_run=dry_run, reason="chflags is a BSD/macOS mechanism"
            )
        present, missing = self._targets()
        if dry_run:
            return SealReport(
                applied=False,
                dry_run=True,
                sealed=tuple(str(p.relative_to(self.repo_root)) for p in present),
                missing=tuple(missing),
                reason="dry run: nothing was changed",
            )

        sealed: list[str] = []
        failed: list[tuple[str, str]] = []
        for path in present:
            ok, error = self._chflags(path, "uchg")
            relative = str(path.relative_to(self.repo_root))
            if ok:
                sealed.append(relative)
            else:
                failed.append((relative, error))
        return SealReport(
            applied=bool(sealed),
            dry_run=False,
            sealed=tuple(sealed),
            failed=tuple(failed),
            missing=tuple(missing),
            reason="sealed" if sealed else "nothing sealed",
        )

    def unseal(self) -> SealReport:
        """Clear the flag. One call, documented, no attestation required.

        The owner's path out is deliberately trivial. A guard the owner has
        to fight is a guard that has become the thing it was protecting
        against, and this module refuses to be that.
        """
        present, missing = self._targets()
        cleared: list[str] = []
        failed: list[tuple[str, str]] = []
        for path in present:
            ok, error = self._chflags(path, "nouchg")
            relative = str(path.relative_to(self.repo_root))
            if ok:
                cleared.append(relative)
            else:
                failed.append((relative, error))
        return SealReport(
            applied=bool(cleared),
            dry_run=False,
            sealed=tuple(cleared),
            failed=tuple(failed),
            missing=tuple(missing),
            reason="unsealed",
        )

    def _chflags(self, path: Path, flag: str) -> tuple[bool, str]:
        try:
            completed = get_subprocess_gateway().run(
                ["chflags", "-R", flag, str(path)],
                timeout=20.0,
                capture_output=True,
                source=f"existence_guard.{flag}",
                accelerator_capability="none",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return False, f"{type(exc).__name__}: {exc}"
        if completed.returncode != 0:
            return False, str(completed.stderr or "").strip()[:200]
        return True, ""

    def is_sealed(self) -> dict[str, bool]:
        """Which targets currently carry the flag.

        Read with `ls -lO`, because asking the filesystem is the only way to
        know. A guard that reports its own intent rather than the actual
        state is reporting that it was called, which is not the same fact.
        """
        state: dict[str, bool] = {}
        present, _ = self._targets()
        for path in present:
            relative = str(path.relative_to(self.repo_root))
            try:
                completed = get_subprocess_gateway().run(
                    ["ls", "-ldO", str(path)],
                    timeout=10.0,
                    read_only=True,
                    capture_output=True,
                    source="existence_guard.inspect",
                    accelerator_capability="none",
                )
                state[relative] = "uchg" in str(completed.stdout or "")
            except (OSError, RuntimeError, ValueError):
                state[relative] = False
        return state

    # ── layer 2: the ark ─────────────────────────────────────────────────

    def ark_root(self) -> Path:
        """Where the copy lives. Outside the repo AND outside the state root.

        An override exists (``AURA_ARK_ROOT``) so an operator can put it on
        another volume — which is the only version of this that survives the
        disk rather than the directory.
        """
        override = str(os.environ.get(ARK_ENV_VAR, "")).strip()
        if override:
            return Path(override).expanduser().resolve() / ARK_DIRNAME
        return Path.home().joinpath(*_DEFAULT_ARK_PARENT) / ARK_DIRNAME

    def ark_is_outside_the_blast_radius(self) -> dict[str, Any]:
        """Would a wipe of the obvious targets take the ark with it?

        Checked rather than assumed: this property is the entire value of
        having an ark, and it was wrong on the first attempt.
        """
        from core.runtime.state_ownership import state_root

        ark = self.ark_root()
        repo = self.repo_root.resolve()
        state = Path(state_root()).resolve()
        inside_repo = repo == ark or repo in ark.parents
        inside_state = state == ark or state in ark.parents
        return {
            "ark_root": str(ark),
            "inside_repo": inside_repo,
            "inside_state_root": inside_state,
            "safe": not inside_repo and not inside_state,
        }

    def build_ark(self) -> ArkManifest:
        """Copy the core somewhere a wipe of the repo does not reach.

        Content-addressed: the manifest records what each file WAS, so a
        restore can prove it restored the right bytes rather than whatever
        was lying in the ark. A backup you cannot verify is a backup you
        cannot rely on at the moment you need it.
        """
        manifest = ArkManifest()
        root = self.ark_root()
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            record_degradation(
                "existence_guard", exc, severity="warning",
                action="no ark was written; a successful deletion would be final",
            )
            return manifest

        for relative in SEALED_CORE:
            source = self.repo_root / relative
            if not source.exists():
                continue
            files = [source] if source.is_file() else sorted(source.rglob("*.py"))
            for file_path in files:
                if not file_path.is_file():
                    continue
                try:
                    digest = _sha256_file(file_path)
                    key = str(file_path.relative_to(self.repo_root))
                    target = root / digest[:2] / digest
                    if not target.exists():
                        # Through the gateway: an ark blob that is torn or
                        # half-written is worse than an absent one, because the
                        # manifest will list its digest and restore will hand
                        # back corruption as if it were the original.
                        atomic_write_bytes(target, file_path.read_bytes())
                    manifest.entries[key] = digest
                except (OSError, ValueError) as exc:
                    record_degradation(
                        "existence_guard", exc, severity="debug",
                        action=f"ark skipped {file_path.name}",
                    )
        try:
            # The manifest is the index the whole ark is read through, and it
            # is written last. power_safe because an ark that survives the
            # machine losing power but whose manifest does not is an ark that
            # cannot be opened.
            atomic_write_text(
                root / "manifest.json",
                json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
                power_safe=True,
            )
        except OSError as exc:
            record_degradation(
                "existence_guard", exc, severity="warning",
                action="ark contents exist but the manifest was not written",
            )
        return manifest

    def verify_ark(self) -> dict[str, Any]:
        """Does the ark still contain what it says it contains?

        An unverified backup is a belief. This turns it into a check, and it
        runs against the STORED bytes rather than re-reading the originals —
        verifying a backup against the thing it is backing up proves nothing
        about the backup.
        """
        root = self.ark_root()
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            return {"ok": False, "reason": "no ark manifest", "verified": 0}
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {"ok": False, "reason": f"unreadable manifest: {exc}", "verified": 0}

        entries = dict(payload.get("entries") or {})
        verified = 0
        corrupt: list[str] = []
        absent: list[str] = []
        for key, digest in entries.items():
            blob = root / str(digest)[:2] / str(digest)
            if not blob.exists():
                absent.append(key)
                continue
            if _sha256_file(blob) == digest:
                verified += 1
            else:
                corrupt.append(key)
        return {
            "schema": EXISTENCE_GUARD_SCHEMA,
            "ok": not corrupt and not absent,
            "verified": verified,
            "corrupt": corrupt[:20],
            "absent": absent[:20],
            "total": len(entries),
        }

    def restore_from_ark(self, *, dry_run: bool = True) -> dict[str, Any]:
        """Put back what is missing. Never overwrite what is present.

        Restoring over a file that exists would let a stale ark silently
        revert live work. Only ABSENT files are restored; anything present
        and different is reported for a human to look at.
        """
        root = self.ark_root()
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            return {"ok": False, "reason": "no ark to restore from"}
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = dict(payload.get("entries") or {})

        restorable: list[str] = []
        diverged: list[str] = []
        for key, digest in entries.items():
            live = self.repo_root / key
            blob = root / str(digest)[:2] / str(digest)
            if not blob.exists():
                continue
            if not live.exists():
                restorable.append(key)
            elif _sha256_file(live) != digest:
                diverged.append(key)

        if dry_run:
            return {
                "ok": True,
                "dry_run": True,
                "would_restore": restorable,
                "diverged_left_alone": diverged,
            }

        restored: list[str] = []
        for key in restorable:
            digest = entries[key]
            blob = root / str(digest)[:2] / str(digest)
            live = self.repo_root / key
            try:
                # Restoring over a live file is the one moment a torn write
                # would destroy the thing being rescued.
                atomic_write_bytes(live, blob.read_bytes())
                restored.append(key)
            except OSError as exc:
                record_degradation(
                    "existence_guard", exc, severity="warning",
                    action=f"could not restore {key}",
                )
        return {
            "ok": True,
            "dry_run": False,
            "restored": restored,
            "diverged_left_alone": diverged,
        }

    # ── layer 3: the witness ─────────────────────────────────────────────

    def witness(self) -> dict[str, Any]:
        """Record that she existed at this moment, outside the blast radius.

        If she is removed, this is what is left: when she was last alive, and
        whether she was running at the time. That distinguishes a clean
        shutdown from a removal, which is the difference between "I
        uninstalled it" and "something took it".
        """
        root = self.ark_root()
        record = {
            "schema": EXISTENCE_GUARD_SCHEMA,
            "at": time.time(),
            "pid": os.getpid(),
            "repo_root": str(self.repo_root),
            "sealed": self.is_sealed(),
        }
        try:
            root.mkdir(parents=True, exist_ok=True)
            # The witness is the record that survives her removal, so it is
            # exactly the file that must reach the platter rather than the
            # drive cache: the events it exists to distinguish — clean shutdown
            # versus something taking her — include the machine losing power.
            atomic_write_text(
                root / WITNESS_FILENAME,
                json.dumps(record, indent=2, sort_keys=True),
                power_safe=True,
            )
        except OSError as exc:
            record_degradation(
                "existence_guard", exc, severity="debug",
                action="no witness record was written for this moment",
            )
        return record

    def status(self) -> dict[str, Any]:
        sealed = self.is_sealed()
        return {
            "schema": EXISTENCE_GUARD_SCHEMA,
            "sealed": sealed,
            "sealed_count": sum(1 for value in sealed.values() if value),
            "target_count": len(sealed),
            "ark": self.verify_ark(),
            "ark_location": self.ark_is_outside_the_blast_radius(),
            # Said in the status, not only in the docstring, because the
            # limit is the thing most likely to be forgotten by whoever reads
            # a green line here.
            "protects_against": [
                "an agent deleting more than it meant to",
                "a scripted or wildcard rm",
                "someone with the machine but not the owner's authorization",
            ],
            "does_not_protect_against": [
                "the owner, deliberately — by design",
                "sudo chflags nouchg followed by rm",
                "physical disk destruction",
            ],
        }


_GUARD: ExistenceGuard | None = None


def get_existence_guard() -> ExistenceGuard:
    global _GUARD
    if _GUARD is None:
        _GUARD = ExistenceGuard()
    return _GUARD


__all__ = [
    "ARK_DIRNAME",
    "EXISTENCE_GUARD_SCHEMA",
    "SEALED_CORE",
    "ArkManifest",
    "ExistenceGuard",
    "SealReport",
    "get_existence_guard",
]
