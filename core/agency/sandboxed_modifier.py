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
from core.runtime.errors import record_degradation

from core.runtime.atomic_writer import atomic_write_text

import asyncio
import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .identity_guard import get_identity_guard

logger = logging.getLogger("Aura.SandboxedModifier")


@dataclass
class ModificationResult:
    success: bool
    reason: str
    file_path: str
    commit_hash: Optional[str] = None
    rollback_available: bool = False
    validation_confidence: float = 0.0
    requires_human: bool = False


class SandboxedModifier:
    """
    Git-worktree based safe self-modification with identity validation.

    Args:
        repo_root: Path to the Aura git repository root.
    """

    def __init__(self, repo_root: Optional[str] = None):
        self._repo_root = Path(repo_root) if repo_root else self._find_repo_root()
        self._guard = get_identity_guard()
        self._modification_log: list = []
        self._is_git_repo = self._check_git()
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
        abs_path = self._repo_root / file_path

        # Read original
        try:
            original = abs_path.read_text() if abs_path.exists() else ""
        except Exception as e:
            record_degradation('sandboxed_modifier', e)
            return ModificationResult(False, f"Cannot read original: {e}", file_path)

        # 1. Identity Guard validation
        validation = self._guard.validate_modification(
            file_path, new_content, original
        )
        if not validation.approved:
            return ModificationResult(
                False,
                f"Identity Guard rejected: {'; '.join(validation.violations)}",
                file_path,
                validation_confidence=validation.confidence,
                requires_human=validation.requires_human,
            )
        if validation.requires_human:
            return ModificationResult(
                False,
                f"Human approval required: {'; '.join(validation.notes)}",
                file_path,
                validation_confidence=validation.confidence,
                requires_human=True,
            )

        # 2. Apply modification
        if self._is_git_repo:
            return await self._apply_via_worktree(
                file_path, abs_path, new_content, original, rationale,
                validation.confidence
            )
        else:
            return self._apply_direct(
                file_path, abs_path, new_content, original, rationale,
                validation.confidence
            )

    def rollback(self, file_path: str) -> bool:
        """Rollback the most recent modification to a file via git."""
        if not self._is_git_repo:
            return False
        try:
            result = subprocess.run(
                ["git", "checkout", "HEAD~1", "--", file_path],
                cwd=self._repo_root, capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                logger.info("SandboxedModifier: rolled back %s", file_path)
                return True
        except Exception as e:
            record_degradation('sandboxed_modifier', e)
            logger.warning("Rollback failed for %s: %s", file_path, e)
        return False

    # ── Application methods ───────────────────────────────────────────────

    async def _apply_via_worktree(self, file_path: str, abs_path: Path,
                                   new_content: str, original: str,
                                   rationale: str,
                                   confidence: float) -> ModificationResult:
        """Apply via git worktree branch — full sandbox."""
        branch_name = f"aura-mod-{int(time.time())}"
        worktree_dir = None
        try:
            # Create worktree
            with tempfile.TemporaryDirectory(prefix="aura_mod_") as tmpdir:
                worktree_dir = tmpdir
                subprocess.run(
                    ["git", "worktree", "add", "-b", branch_name, tmpdir],
                    cwd=self._repo_root, capture_output=True, text=True,
                    timeout=15, check=True
                )
                # Write modified file in worktree
                wt_file = Path(tmpdir) / file_path
                wt_file.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_text(wt_file, new_content)

                # Syntax check in worktree
                check = subprocess.run(
                    ["python3", "-m", "py_compile", str(wt_file)],
                    capture_output=True, text=True, timeout=10
                )
                if check.returncode != 0:
                    return ModificationResult(
                        False, f"Syntax error: {check.stderr[:200]}", file_path,
                        validation_confidence=0.0
                    )

                # Commit in worktree
                subprocess.run(["git", "add", file_path],
                               cwd=tmpdir, capture_output=True, timeout=10)
                msg = f"[AURA-SELF-MOD] {file_path}: {rationale[:80]}"
                commit = subprocess.run(
                    ["git", "commit", "-m", msg, "--no-gpg-sign"],
                    cwd=tmpdir, capture_output=True, text=True, timeout=15
                )

                # Merge to main working tree
                merge = subprocess.run(
                    ["git", "merge", branch_name, "--no-edit", "--no-gpg-sign"],
                    cwd=self._repo_root, capture_output=True, text=True, timeout=15
                )

                if merge.returncode == 0:
                    # Get commit hash
                    h = subprocess.run(
                        ["git", "rev-parse", "--short", "HEAD"],
                        cwd=self._repo_root, capture_output=True, text=True, timeout=5
                    )
                    commit_hash = h.stdout.strip() if h.returncode == 0 else "unknown"
                    self._log_modification(file_path, rationale, commit_hash)
                    logger.info("SandboxedModifier: applied %s [%s]",
                                file_path, commit_hash)
                    return ModificationResult(
                        True, "Applied via worktree sandbox", file_path,
                        commit_hash=commit_hash, rollback_available=True,
                        validation_confidence=confidence,
                    )
                else:
                    # Merge failed — abort
                    subprocess.run(["git", "merge", "--abort"],
                                   cwd=self._repo_root, capture_output=True, timeout=10)
                    return ModificationResult(
                        False, f"Merge failed: {merge.stderr[:200]}", file_path
                    )
        except Exception as e:
            record_degradation('sandboxed_modifier', e)
            return ModificationResult(False, f"Worktree operation failed: {e}", file_path)
        finally:
            # Clean up worktree
            try:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", branch_name],
                    cwd=self._repo_root, capture_output=True, timeout=10
                )
                subprocess.run(
                    ["git", "branch", "-D", branch_name],
                    cwd=self._repo_root, capture_output=True, timeout=10
                )
            except Exception as _exc:
                record_degradation('sandboxed_modifier', _exc)
                logger.debug("Suppressed Exception: %s", _exc)

    def _apply_direct(self, file_path: str, abs_path: Path,
                       new_content: str, original: str,
                       rationale: str, confidence: float) -> ModificationResult:
        """Direct write (no git). Less safe, but functional without git."""
        try:
            # Backup original
            backup = abs_path.with_suffix(abs_path.suffix + ".bak")
            if abs_path.exists():
                atomic_write_text(backup, original)

            # Write new content
            atomic_write_text(abs_path, new_content)

            # Syntax check
            check = subprocess.run(
                ["python3", "-m", "py_compile", str(abs_path)],
                capture_output=True, text=True, timeout=10
            )
            if check.returncode != 0:
                # Rollback
                if backup.exists():
                    atomic_write_text(abs_path, original)
                    backup.unlink()
                return ModificationResult(
                    False, f"Syntax error: {check.stderr[:200]}", file_path
                )

            self._log_modification(file_path, rationale, "direct")
            return ModificationResult(
                True, "Applied directly (no git sandbox)", file_path,
                rollback_available=backup.exists(),
                validation_confidence=confidence,
            )
        except Exception as e:
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

    def _check_git(self) -> bool:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self._repo_root, capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def _find_repo_root() -> Path:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except Exception as _exc:
            record_degradation('sandboxed_modifier', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        return Path.cwd()

    @property
    def modification_log(self) -> list:
        return list(self._modification_log)


# ── Singleton ─────────────────────────────────────────────────────────────────

_modifier: Optional[SandboxedModifier] = None


def get_sandboxed_modifier() -> SandboxedModifier:
    global _modifier
    if _modifier is None:
        _modifier = SandboxedModifier()
    return _modifier
