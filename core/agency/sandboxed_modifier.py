"""core/agency/sandboxed_modifier.py
Sandboxed Self-Modifier
=========================
Enables Aura to safely modify her own source code using git worktrees
as isolated sandboxes. Every modification is:

  1. Made in an isolated branch (worktree)
  2. Validated by IdentityGuard before merging
  3. Syntax-checked before merging
  4. Immediately reversible (git checkout rollback)
  5. Logged with rationale

Workflow:
  sandbox = SandboxedModifier("/path/to/aura")
  result = await sandbox.modify(file_path, new_content, rationale)
  if result.success:
      logger.info("Modification applied: %s", result.commit_hash)
  else:
      logger.warning("Modification rejected: %s", result.reason)

This is NOT general code execution. It is ONLY for modifying
Aura's own modules with full audit trail and rollback contract.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.tasks.managed_command import ManagedCommandResult

from .identity_guard import get_identity_guard

logger = logging.getLogger("Aura.SandboxedModifier")

_RECOVERABLE_MODIFIER_ERRORS = (
    FileNotFoundError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
CommandRunner = Callable[[tuple[str, ...], Path, float], Awaitable[ManagedCommandResult]]


@dataclass
class ModificationResult:
    success: bool
    reason: str
    file_path: str
    commit_hash: str | None = None
    rollback_available: bool = False
    validation_confidence: float = 0.0
    requires_human: bool = False


async def _default_command_runner(
    command: tuple[str, ...],
    cwd: Path,
    timeout_s: float,
) -> ManagedCommandResult:
    started = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        stdout=-1,
        stderr=-1,
    )
    timed_out = False
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except TimeoutError:
        timed_out = True
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()

    elapsed = time.perf_counter() - started
    return ManagedCommandResult(
        command,
        process.returncode,
        stdout_bytes.decode("utf-8", errors="replace"),
        stderr_bytes.decode("utf-8", errors="replace"),
        elapsed,
        timed_out=timed_out,
    )


class SandboxedModifier:
    """
    Git-worktree based safe self-modification with identity validation.

    Args:
        repo_root: Path to the Aura git repository root.
    """

    def __init__(
        self,
        repo_root: str | None = None,
        *,
        command_runner: CommandRunner | None = None,
        guard: object | None = None,
    ):
        self._repo_root = (Path(repo_root) if repo_root else self._find_repo_root()).resolve()
        self._guard = guard or get_identity_guard()
        self._command_runner = command_runner or _default_command_runner
        self._modification_log: list = []
        self._is_git_repo = self._looks_like_git_repo()
        if not self._is_git_repo:
            logger.warning("SandboxedModifier: not a git repo — modifications will be "
                           "direct (no sandbox). Rollback unavailable.")
        logger.info("SandboxedModifier online (git=%s, root=%s)",
                    self._is_git_repo, self._repo_root)

    # ── Public API ────────────────────────────────────────────────────────

    async def modify(self, file_path: str, new_content: str,
                     rationale: str = "") -> ModificationResult:
        """
        Apply a modification to a source file with full safety pipeline:
        identity validation → syntax check → sandboxed apply → merge.

        file_path: relative path from repo root (e.g. "core/brain/inference_gate.py")
        new_content: the complete new file content
        rationale: why this modification is being made (logged)
        """
        try:
            relative_path, abs_path = self._resolve_target(file_path)
        except _RECOVERABLE_MODIFIER_ERRORS as e:
            record_degradation("sandboxed_modifier", e)
            return ModificationResult(False, f"Invalid target path: {e}", file_path)

        # Read original
        try:
            original = abs_path.read_text() if abs_path.exists() else ""
        except _RECOVERABLE_MODIFIER_ERRORS as e:
            record_degradation('sandboxed_modifier', e)
            return ModificationResult(False, f"Cannot read original: {e}", relative_path)

        if original == new_content:
            return ModificationResult(
                True,
                "No change needed",
                relative_path,
                rollback_available=False,
                validation_confidence=1.0,
            )

        # 1. Identity Guard validation
        validation = self._guard.validate_modification(
            relative_path, new_content, original
        )
        if not validation.approved:
            return ModificationResult(
                False,
                f"Identity Guard rejected: {'; '.join(validation.violations)}",
                relative_path,
                validation_confidence=validation.confidence,
                requires_human=validation.requires_human,
            )
        if validation.requires_human:
            return ModificationResult(
                False,
                f"Human approval required: {'; '.join(validation.notes)}",
                relative_path,
                validation_confidence=validation.confidence,
                requires_human=True,
            )

        # 2. Apply modification
        if self._is_git_repo:
            return await self._apply_via_worktree(
                relative_path, abs_path, new_content, original, rationale,
                validation.confidence
            )
        else:
            return self._apply_direct(
                relative_path, abs_path, new_content, original, rationale,
                validation.confidence
            )

    def rollback(self, file_path: str) -> bool:
        """Rollback the most recent modification to a file via git."""
        try:
            relative_path, _ = self._resolve_target(file_path)
            if not self._is_git_repo:
                return self._rollback_direct(relative_path)
            result = self._run_command_blocking(
                ("git", "checkout", "HEAD~1", "--", relative_path),
                self._repo_root,
                10.0,
            )
            if result.ok:
                logger.info("SandboxedModifier: rolled back %s", relative_path)
                return True
        except _RECOVERABLE_MODIFIER_ERRORS as e:
            record_degradation('sandboxed_modifier', e)
            logger.warning("Rollback failed for %s: %s", file_path, e)
        return False

    # ── Application methods ───────────────────────────────────────────────

    async def _apply_via_worktree(self, file_path: str, abs_path: Path,
                                   new_content: str, original: str,
                                   rationale: str,
                                   confidence: float) -> ModificationResult:
        """Apply via git worktree branch — full sandbox."""
        branch_name = f"aura-mod-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        try:
            # Create worktree
            with tempfile.TemporaryDirectory(prefix="aura_mod_root_") as tmpdir:
                worktree_path = Path(tmpdir) / "worktree"
                worktree = await self._run_command(
                    ("git", "worktree", "add", "-b", branch_name, str(worktree_path), "HEAD"),
                    self._repo_root,
                    15.0,
                )
                if not worktree.ok:
                    return ModificationResult(
                        False,
                        self._command_failure("Worktree create failed", worktree),
                        file_path,
                        validation_confidence=confidence,
                    )

                # Write modified file in worktree
                wt_file = worktree_path / file_path
                wt_file.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_text(wt_file, new_content)

                # Syntax check in worktree
                check = await self._run_command(
                    (sys.executable, "-m", "py_compile", str(wt_file)),
                    worktree_path,
                    10.0,
                )
                if not check.ok:
                    return ModificationResult(
                        False,
                        self._command_failure("Syntax check failed", check),
                        file_path,
                        validation_confidence=0.0,
                    )

                # Commit in worktree
                add = await self._run_command(("git", "add", file_path), worktree_path, 10.0)
                if not add.ok:
                    return ModificationResult(
                        False,
                        self._command_failure("Git add failed", add),
                        file_path,
                        validation_confidence=confidence,
                    )
                msg = f"[AURA-SELF-MOD] {file_path}: {rationale[:80]}"
                commit = await self._run_command(
                    ("git", "commit", "-m", msg, "--no-gpg-sign"),
                    worktree_path,
                    15.0,
                )
                if not commit.ok:
                    return ModificationResult(
                        False,
                        self._command_failure("Commit failed", commit),
                        file_path,
                        validation_confidence=confidence,
                    )

                # Merge to main working tree
                merge = await self._run_command(
                    ("git", "merge", branch_name, "--no-edit", "--no-gpg-sign"),
                    self._repo_root,
                    15.0,
                )

                if merge.ok:
                    # Get commit hash
                    h = await self._run_command(
                        ("git", "rev-parse", "--short", "HEAD"),
                        self._repo_root,
                        5.0,
                    )
                    commit_hash = h.stdout.strip() if h.ok else "unknown"
                    self._log_modification(file_path, rationale, commit_hash)
                    logger.info("SandboxedModifier: applied %s [%s]",
                                file_path, commit_hash)
                    return ModificationResult(
                        True, "Applied via worktree sandbox", file_path,
                        commit_hash=commit_hash, rollback_available=True,
                        validation_confidence=confidence,
                    )
                # Merge failed — abort before removing the branch.
                await self._run_command(("git", "merge", "--abort"), self._repo_root, 10.0)
                return ModificationResult(
                    False,
                    self._command_failure("Merge failed", merge),
                    file_path,
                    validation_confidence=confidence,
                )
        except _RECOVERABLE_MODIFIER_ERRORS as e:
            record_degradation('sandboxed_modifier', e)
            return ModificationResult(False, f"Worktree operation failed: {e}", file_path)
        finally:
            # Clean up worktree
            try:
                if "worktree_path" in locals():
                    await self._run_command(
                        ("git", "worktree", "remove", "--force", str(worktree_path)),
                        self._repo_root,
                        10.0,
                    )
                await self._run_command(("git", "branch", "-D", branch_name), self._repo_root, 10.0)
            except _RECOVERABLE_MODIFIER_ERRORS as _exc:
                record_degradation('sandboxed_modifier', _exc)
                logger.debug("Worktree cleanup failed: %s", _exc)

    def _apply_direct(self, file_path: str, abs_path: Path,
                       new_content: str, original: str,
                       rationale: str, confidence: float) -> ModificationResult:
        """Direct write (no git). Less safe, but functional without git."""
        try:
            # Backup original
            existed_before = abs_path.exists()
            backup = abs_path.with_suffix(abs_path.suffix + ".bak")
            if existed_before:
                atomic_write_text(backup, original)

            # Write new content
            atomic_write_text(abs_path, new_content)

            # Syntax check
            check = self._run_command_blocking(
                (sys.executable, "-m", "py_compile", str(abs_path)),
                self._repo_root,
                10.0,
            )
            if not check.ok:
                # Rollback
                if backup.exists():
                    atomic_write_text(abs_path, original)
                    backup.unlink()
                elif not existed_before and abs_path.exists():
                    abs_path.unlink()
                return ModificationResult(
                    False,
                    self._command_failure("Syntax check failed", check),
                    file_path,
                )

            self._log_modification(file_path, rationale, "direct")
            return ModificationResult(
                True, "Applied directly (no git sandbox)", file_path,
                rollback_available=backup.exists(),
                validation_confidence=confidence,
            )
        except _RECOVERABLE_MODIFIER_ERRORS as e:
            record_degradation('sandboxed_modifier', e)
            return ModificationResult(False, f"Direct write failed: {e}", file_path)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _log_modification(self, file_path: str, rationale: str, ref: str):
        self._modification_log.append({
            "file": file_path,
            "rationale": rationale,
            "ref": ref,
            "timestamp": time.time(),
        })
        if len(self._modification_log) > 100:
            self._modification_log = self._modification_log[-100:]

    def _resolve_target(self, file_path: str) -> tuple[str, Path]:
        requested = Path(file_path)
        if requested.is_absolute():
            raise ValueError("self-modification target must be relative to the repo root")

        resolved = (self._repo_root / requested).resolve()
        try:
            relative = resolved.relative_to(self._repo_root)
        except ValueError as exc:
            raise ValueError("self-modification target must stay inside the repo root") from exc

        if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("self-modification target contains invalid path segments")
        return relative.as_posix(), resolved

    def _looks_like_git_repo(self) -> bool:
        return (self._repo_root / ".git").exists()

    def _rollback_direct(self, file_path: str) -> bool:
        target = self._repo_root / file_path
        backup = target.with_suffix(target.suffix + ".bak")
        if not backup.exists():
            return False
        original = backup.read_text()
        atomic_write_text(target, original)
        backup.unlink()
        logger.info("SandboxedModifier: directly rolled back %s", file_path)
        return True

    @staticmethod
    def _find_repo_root() -> Path:
        current = Path.cwd().resolve()
        for candidate in (current, *current.parents):
            if (candidate / ".git").exists():
                return candidate
        return current

    async def _run_command(
        self,
        command: tuple[str, ...],
        cwd: Path,
        timeout_s: float,
    ) -> ManagedCommandResult:
        try:
            return await self._command_runner(command, cwd, timeout_s)
        except _RECOVERABLE_MODIFIER_ERRORS as exc:
            record_degradation("sandboxed_modifier", exc)
            logger.warning("SandboxedModifier command failed before launch: %s", exc)
            return ManagedCommandResult(command, 127, "", str(exc), 0.0)

    def _run_command_blocking(
        self,
        command: tuple[str, ...],
        cwd: Path,
        timeout_s: float,
    ) -> ManagedCommandResult:
        def command_call() -> Awaitable[ManagedCommandResult]:
            return self._run_command(command, cwd, timeout_s)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(command_call())

        result: dict[str, ManagedCommandResult] = {}
        failure: list[BaseException] = []

        def runner() -> None:
            try:
                result["value"] = asyncio.run(command_call())
            except _RECOVERABLE_MODIFIER_ERRORS as exc:
                failure.append(exc)

        thread = threading.Thread(target=runner, name="aura-sandboxed-modifier-command", daemon=True)
        thread.start()
        thread.join()
        if failure:
            exc = failure[0]
            record_degradation("sandboxed_modifier", exc)
            return ManagedCommandResult(command, 127, "", str(exc), 0.0)
        return result["value"]

    @staticmethod
    def _command_failure(prefix: str, result: ManagedCommandResult) -> str:
        if result.timed_out:
            return f"{prefix}: timed out after {result.elapsed_s:.1f}s"
        detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
        return f"{prefix}: {detail[:300]}"

    @property
    def modification_log(self) -> list:
        return list(self._modification_log)


# ── Singleton ─────────────────────────────────────────────────────────────────

_modifier: SandboxedModifier | None = None


def get_sandboxed_modifier() -> SandboxedModifier:
    global _modifier
    if _modifier is None:
        _modifier = SandboxedModifier()
    return _modifier
