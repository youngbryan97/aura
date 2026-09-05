"""Safe Self-Modification System
Version control integration with automatic rollback on failure.

v5.2: Added path allowlisting, risk gating, backup integrity verification,
      and event bus integration for modification proposals.
"""

import ast
import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import config
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.subprocess_gateway import get_subprocess_gateway

from .boot_validator import GhostBootValidator
from .mutation_constitution import DEFER, PROPOSE, REFUSE, admit_mutation
from .mutation_tiers import MutationTier, classify_mutation_path
from .promotion_policy import (
    SAFE_AUTONOMOUS_REPAIR_ENV as _SAFE_AUTONOMOUS_REPAIR_ENV,
    SUPERVISED_SELF_MODIFICATION_ENV as _SUPERVISED_SELF_MODIFICATION_ENV,
)
from .promotion_policy import env_flag, safe_autonomous_repair_decision, source_promotion_decision
from .safe_modification_harness import SafeModificationHarness

logger = logging.getLogger("SelfModification.SafeModification")


@dataclass
class ModificationRecord:
    """Record of a self-modification attempt"""

    timestamp: float
    file_path: str
    fix_description: str
    success: bool
    commit_hash: str | None = None
    error: str | None = None
    test_results: dict[str, Any] | None = None

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "file_path": self.file_path,
            "fix_description": self.fix_description,
            "success": self.success,
            "commit_hash": self.commit_hash,
            "error": self.error,
            "test_results": self.test_results,
        }


@dataclass
class LogicTransplant:
    """Represents a multi-block architectural shift."""

    target_file: str
    explanation: str
    chunks: list[dict[str, str]]  # List of {"original": "...", "fixed": "..."}
    risk_level: int = 5
    lines_changed: int = 0

    def to_dict(self):
        return {
            "target_file": self.target_file,
            "explanation": self.explanation,
            "chunks": self.chunks,
            "risk_level": self.risk_level,
            "lines_changed": self.lines_changed,
        }


class GitIntegration:
    """Git version control integration for safe self-modification."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.auto_commit_enabled = True

        # Verify git is available (Sync check for initialization)
        self.git_available = self._check_git_available_sync()
        if not self.git_available:
            logger.warning("Git not available - working without version control")
        else:
            logger.info("Git integration initialized for %s", self.repo_path)

    def _check_git_available_sync(self) -> bool:
        """Synchronous check for git availability."""
        if shutil.which("git") is None:
            return False
        try:
            result = get_subprocess_gateway().run(
                ["git", "status"],
                cwd=self.repo_path,
                capture_output=True,
                read_only=True,
                timeout=5,
                source="self_modification.safe_modification.git_status",
                accelerator_capability="none",
            )
            return result.returncode == 0
        except (OSError, RuntimeError, TimeoutError, ValueError):
            return False

    @staticmethod
    def _validate_path(file_path: str) -> str:
        """Issue 73/74: Validate and sanitize file paths (Link traversal & Unicode)."""
        import re

        # Block non-ASCII (Unicode path check - Issue 74)
        if not all(ord(c) < 128 for c in file_path):
            raise ValueError(f"Unicode characters not allowed in file paths: {file_path!r}")

        # Block shell metacharacters
        if re.search(r"[;&|`$(){}\[\]<>!\\\n\r]", file_path):
            raise ValueError(f"Path contains shell metacharacters: {file_path!r}")

        # Block path traversal (Issue 73)
        if ".." in file_path:
            raise ValueError(f"Path traversal detected: {file_path!r}")

        # Link traversal check (Issue 73)
        try:
            full_path = os.path.realpath(file_path)
            cwd_path = os.path.realpath(os.getcwd())
            if not full_path.startswith(cwd_path):
                raise ValueError(f"Path escaped project root via symlink: {file_path!r}")
        except OSError as e:
            record_degradation("safe_modification", e)
            if isinstance(e, ValueError):
                raise
            # If path doesn't exist yet, we can't realpath it fully, but we checked ..
            logger.debug("Path validation exception (likely non-existent): %s", e)

        # Block absolute paths outside the repo
        if file_path.startswith("/"):
            raise ValueError(f"Absolute paths not allowed: {file_path!r}")

        # Must be a Python file or known config
        if not file_path.endswith((".py", ".toml", ".yaml", ".yml", ".json", ".cfg")):
            raise ValueError(f"Unsupported file type: {file_path!r}")
        return file_path

    @staticmethod
    def _validate_branch_name(branch_name: str) -> str:
        """C-04 FIX: Validate branch names to prevent injection."""
        import re

        if not re.match(r"^[a-zA-Z0-9_\-/\.]+$", branch_name):
            raise ValueError(f"Invalid branch name: {branch_name!r}")
        if len(branch_name) > 100:
            raise ValueError(f"Branch name too long: {len(branch_name)}")
        return branch_name

    async def _check_git_available(self) -> bool:
        """Check if git is installed and repo exists (Async)"""
        return await asyncio.to_thread(self._check_git_available_sync)

    async def _git(
        self,
        argv: list[str],
        *,
        timeout: int = 10,
        source: str = "",
        read_only: bool = False,
    ):
        """Every git invocation this class makes, through one door.

        There were eleven separate gateway calls here, each repeating the
        working directory, the capture flag and the accelerator declaration —
        so adding a check to how Aura runs git against her own repository
        meant finding and editing eleven sites, and missing one meant the
        check did not apply.
        """
        return await get_subprocess_gateway().run_async(
            argv,
            cwd=self.repo_path,
            capture_output=True,
            timeout=timeout,
            read_only=read_only,
            source=source or "core.self_modification.safe_modification.git",
            accelerator_capability="none",
        )

    async def is_worktree_dirty(self) -> bool:
        """Return True when the repository has any local changes."""
        if not self.git_available:
            return False
        try:
            result = await self._git(
                ["git", "status", "--porcelain"],
                timeout=5,
                read_only=True,
                source="core.self_modification.safe_modification.git_status",
            )
            return bool(result.stdout.strip())
        except (subprocess.SubprocessError, OSError) as e:
            record_degradation("safe_modification", e)
            logger.debug("Dirty worktree check failed: %s", e)
            return True

    async def create_branch(self, branch_name: str) -> bool:
        """Create and checkout a new branch for testing fix (Async)."""
        if not self.git_available:
            logger.warning("Git not available, skipping branch creation")
            return False
        if await self.is_worktree_dirty():
            logger.info(
                "Git worktree is dirty; skipping branch creation for safe autonomous fix flow."
            )
            return False

        try:
            branch_name = self._validate_branch_name(branch_name)
            # Create branch
            result = await self._git(
                ["git", "checkout", "-b", branch_name],
                timeout=10,
                source="core.self_modification.safe_modification.create_branch",
            )

            if result.returncode == 0:
                logger.info("Created branch: %s", branch_name)
                return True
            else:
                logger.error("Branch creation failed: %s", result.stderr)
                return False

        except (subprocess.SubprocessError, OSError) as e:
            record_degradation("safe_modification", e)
            logger.error("Branch creation exception: %s", e)
            return False

    async def commit_changes(self, file_path: str, message: str) -> str | None:
        """Commit changes to current branch (Async)."""
        if not self.git_available:
            return None

        try:
            file_path = self._validate_path(file_path)
            # Stage file
            await self._git(
                ["git", "add", file_path],
                check=True,
                timeout=5,
                source="core.self_modification.safe_modification.git_add",
            )

            # Commit
            result = await self._git(
                ["git", "commit", "-m", message],
                timeout=10,
                source="core.self_modification.safe_modification.git_commit",
            )

            if result.returncode != 0:
                logger.error("Commit failed: %s", result.stderr)
                return None

            # Get commit hash
            hash_result = await self._git(
                ["git", "rev-parse", "HEAD"],
                timeout=5,
                read_only=True,
                source="core.self_modification.safe_modification.rev_parse_head",
            )

            commit_hash = hash_result.stdout.strip()
            logger.info("Committed changes: %s", commit_hash[:8])
            return commit_hash

        except (subprocess.SubprocessError, OSError) as e:
            record_degradation("safe_modification", e)
            logger.error("Commit exception: %s", e)
            return None

    async def merge_to_main(self, branch_name: str) -> bool:
        """Merge branch into main after successful testing (Async)."""
        if not self.git_available:
            return False

        try:
            # Checkout main
            await self._git(
                ["git", "checkout", "main"],
                check=True,
                timeout=5,
                source="core.self_modification.safe_modification.checkout_main_for_merge",
            )

            branch_name = self._validate_branch_name(branch_name)
            # Merge
            result = await self._git(
                ["git", "merge", "--no-ff", branch_name, "-m", f"Auto-merge: {branch_name}"],
                timeout=10,
                source="core.self_modification.safe_modification.merge_to_main",
            )

            if result.returncode == 0:
                logger.info("Merged %s into main", branch_name)
                return True
            else:
                logger.error("Merge failed: %s", result.stderr)
                return False

        except (subprocess.SubprocessError, OSError) as e:
            record_degradation("safe_modification", e)
            logger.error("Merge exception: %s", e)
            return False

    async def delete_branch(self, branch_name: str) -> bool:
        """Delete a branch (Async)"""
        if not self.git_available:
            return False

        try:
            branch_name = self._validate_branch_name(branch_name)
            await self._git(
                ["git", "branch", "-D", branch_name],
                check=True,
                timeout=5,
                source="core.self_modification.safe_modification.delete_branch",
            )
            logger.info("Deleted branch: %s", branch_name)
            return True
        except (subprocess.SubprocessError, OSError) as e:
            record_degradation("safe_modification", e)
            logger.error("Branch deletion failed: %s", e)
            return False

    async def checkout_main(self) -> bool:
        """Return to main branch (Async)"""
        if not self.git_available:
            return False

        try:
            await self._git(
                ["git", "checkout", "main"],
                check=True,
                timeout=5,
                source="core.self_modification.safe_modification.checkout_main",
            )
            return True
        except (subprocess.SubprocessError, OSError):
            return False

    async def get_current_branch(self) -> str | None:
        """Get name of current branch (Async)"""
        if not self.git_available:
            return None

        try:
            result = await self._git(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                timeout=5,
                read_only=True,
                source="core.self_modification.safe_modification.current_branch",
            )
            return result.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None


class BackupSystem:
    """File-based backup system for when git is unavailable."""

    def __init__(self, backup_dir: str | None = None):
        if backup_dir is None:
            from core.config import config

            self.backup_dir = config.paths.data_dir / "backups"
        else:
            self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info("BackupSystem initialized at %s", self.backup_dir)

    def create_backup(self, file_path: str) -> str | None:
        """Create backup of file.

        Args:
            file_path: File to backup

        Returns:
            Backup ID or None

        """
        source = Path(file_path)
        if not source.exists():
            logger.error("Cannot backup non-existent file: %s", file_path)
            return None

        # Generate backup ID
        backup_id = f"{int(time.time())}_{source.name}"
        backup_path = self.backup_dir / backup_id

        try:
            shutil.copy2(source, backup_path)
            logger.info("Created backup: %s", backup_id)

            # Store metadata
            metadata = {
                "original_path": str(source),
                "backup_time": time.time(),
                "backup_id": backup_id,
            }

            metadata_path = self.backup_dir / f"{backup_id}.meta"
            get_file_write_gateway().write_text(
                metadata_path,
                json.dumps(metadata),
                source="self_modification.safe_modification.backup_metadata",
            )

            return backup_id

        except OSError as e:
            record_degradation("safe_modification", e)
            logger.error("Backup creation failed: %s", e)
            return None

    def restore_backup(self, backup_id: str) -> bool:
        """Restore file from backup.

        Args:
            backup_id: ID of backup to restore

        Returns:
            True if successful

        """
        backup_path = self.backup_dir / backup_id
        metadata_path = self.backup_dir / f"{backup_id}.meta"

        if not backup_path.exists() or not metadata_path.exists():
            logger.error("Backup not found: %s", backup_id)
            return False

        try:
            # Read metadata
            with open(metadata_path) as f:
                metadata = json.load(f)

            original_path = Path(metadata["original_path"])

            # Restore file
            shutil.copy2(backup_path, original_path)
            logger.info("Restored backup %s to %s", backup_id, original_path)
            return True

        except OSError as e:
            record_degradation("safe_modification", e)
            logger.error("Restore failed: %s", e)
            return False

    def cleanup_old_backups(self, max_age_days: int = 7):
        """Remove backups older than specified days"""
        cutoff = time.time() - (max_age_days * 86400)

        for backup_file in self.backup_dir.glob("*"):
            if backup_file.suffix != ".meta":
                try:
                    # Check age
                    if backup_file.stat().st_mtime < cutoff:
                        backup_file.unlink()
                        # Remove metadata too
                        meta_file = backup_file.with_suffix(backup_file.suffix + ".meta")
                        if meta_file.exists():
                            meta_file.unlink()
                        logger.debug("Cleaned up old backup: %s", backup_file.name)
                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                    record_degradation("safe_modification", e)
                    logger.error("Cleanup failed for %s: %s", backup_file, e)


@dataclass(frozen=True)
class TurnTrust:
    """What is known about this turn's inputs. Unknown is not trusted."""

    state: str  # "trusted" | "untrusted" | "unknown"
    reason: str

    def __post_init__(self) -> None:
        if self.state not in {"trusted", "untrusted", "unknown"}:
            raise ValueError(f"unknown turn-trust state: {self.state!r}")


def _owner_approved(fix: Any) -> bool:
    """Whether the owner explicitly authorized this specific modification."""
    return bool(
        getattr(fix, "owner_approved", False)
        or getattr(fix, "human_approved", False)
        or getattr(fix, "explicit_owner_approval", False)
    )


def _file_hash(path: Path) -> str:
    """SHA-256 hash of a file for integrity verification."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class SafeSelfModification:
    """Orchestrates safe self-modification with multiple safety layers.

    v5.2 Safety Features:
      - Path allowlisting (ALLOWED_PATHS)
      - Risk gating (MAX_RISK_LEVEL, MAX_LINES_CHANGED)
      - Backup integrity verification (SHA-256)
      - Event bus integration for proposals
    """

    def __init__(
        self,
        code_base_path: str = ".",
        modification_log: str | None = None,
        event_bus=None,
    ):
        self.code_base = Path(code_base_path)
        self.git = GitIntegration(code_base_path)
        self.backup = BackupSystem()
        self.event_bus = event_bus  # Optional InputBus for emitting proposals
        self.boot_validator = GhostBootValidator(self.code_base)

        if modification_log is None:
            self.modification_log = config.paths.data_dir / "modifications.jsonl"
        else:
            self.modification_log = Path(modification_log)
        self.modification_log.parent.mkdir(parents=True, exist_ok=True)

        # Staging Directory for Mutation Quarantine
        self.staging_dir = config.paths.data_dir / "mutation_staging"
        self.staging_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            "total_attempts": 0,
            "successful": 0,
            "failed": 0,
            "rolled_back": 0,
            "blocked_by_policy": 0,
        }

        logger.info("SafeSelfModification system initialized")

    # ------------------------------------------------------------------
    # Safety Gating
    # ------------------------------------------------------------------

    @staticmethod
    def is_allowed_path(file_path: str) -> bool:
        """Check if a file path is within the modification allowlist."""
        normalized = str(file_path).replace("\\", "/")
        return any(
            normalized.startswith(prefix) or f"/{prefix}" in normalized
            for prefix in config.modification.allowed_paths
        )

    @staticmethod
    def is_protected_path(file_path: str) -> bool:
        """Check if a file path is inside a constitutionally protected area."""
        normalized = str(file_path).replace("\\", "/").lstrip("./")
        if classify_mutation_path(normalized).tier is MutationTier.SEALED:
            return True
        for prefix in config.modification.protected_paths:
            protected = str(prefix).replace("\\", "/").lstrip("./")
            if normalized == protected or normalized.startswith(protected.rstrip("/") + "/"):
                return True
        return False

    def _resolve_target_path(self, file_path: str | Path) -> Path:
        target = Path(file_path)
        if not target.is_absolute():
            target = self.code_base / target
        resolved = target.resolve()
        code_root = self.code_base.resolve()
        staging_root = self.staging_dir.resolve()
        try:
            resolved.relative_to(code_root)
        except ValueError:
            try:
                resolved.relative_to(staging_root)
            except ValueError as exc:
                raise ValueError(f"Target path escaped code base: {file_path}") from exc
        return resolved

    def _relative_target_path(self, file_path: str | Path) -> str:
        resolved = self._resolve_target_path(file_path)
        return resolved.relative_to(self.code_base.resolve()).as_posix()

    def _turn_trust_verdict(self) -> TurnTrust:
        """Whether this turn can be trusted to propose a self-modification.

        Three answers, not two. This returned "" for both "the turn read
        nothing" and "the check itself broke", so a provenance lookup that
        raised was indistinguishable from a clean turn and the patch went
        through. The argument for that was real — a broken lookup must not
        stop Aura repairing herself on a turn that read nothing — but it makes
        a claim the code did not establish, on the one surface where being
        wrong means she rewrites her own source at a stranger's suggestion.

        Unknown is its own answer, and it defers rather than refuses: the
        proposal keeps its evidence and can be applied on a turn where the
        question can be answered, or with explicit owner approval.
        """
        try:
            from core.security.content_provenance import describe_untrusted_context
            from core.security.rule_of_two import get_rule_of_two_registry

            registry = get_rule_of_two_registry()
            handler = registry.get("self_modification_apply")
            if handler is None:
                # An empty registry is an uninstalled one, not an unknowable
                # turn: the declarations are a static property of the source.
                # Install them and ask again before calling this unknown.
                from core.security.rule_of_two import install_known_handlers

                install_known_handlers()
                handler = registry.get("self_modification_apply")
            if handler is None:
                return TurnTrust(
                    state="unknown",
                    reason=(
                        "rule-of-two declares no handler for "
                        "self_modification_apply even after installation, so "
                        "nothing checked whether this turn read untrusted content"
                    ),
                )
            if not handler.violates_now():
                return TurnTrust(state="trusted", reason="")
            return TurnTrust(
                state="untrusted",
                reason=describe_untrusted_context() or "this turn read untrusted content",
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "safe_modification",
                exc,
                severity="warning",
                action=(
                    "deferred the modification because the untrusted-context "
                    "check could not run; rule-of-two is not covering "
                    "self-modification on this turn"
                ),
                enforce_failure_policy=False,
            )
            return TurnTrust(state="unknown", reason=f"provenance check failed: {exc}")

    def validate_proposal(self, fix) -> tuple[bool, str]:
        """Gate a modification proposal before it reaches apply_fix.

        Returns:
            (allowed, reason)
        """
        if not fix.target_file:
            return False, "No target file specified"

        # Rule of Two, asked of THIS turn. `self_modification_apply` declares
        # TRUSTED input because the patch is model-generated — true until the
        # model has read a web page, a repository or a tool result, at which
        # point "model-generated" means "generated after reading something a
        # stranger wrote". Untrusted input + executes + in-process is three of
        # three, and this is the surface where three legs means Aura rewrites
        # her own source at a stranger's suggestion.
        trust = self._turn_trust_verdict()
        if trust.state == "untrusted":
            self.stats["blocked_by_policy"] += 1
            return False, (
                f"Self-modification refused: {trust.reason}. A patch proposed during "
                "a turn that ingested untrusted content is not a trusted proposal, "
                "however well it tests."
            )

        try:
            normalized_target = self._relative_target_path(fix.target_file)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("safe_modification", exc)
            self.stats["blocked_by_policy"] += 1
            return False, f"Target path resolution failed: {exc}"

        # 1. Path Allowlist Check
        if not self.is_allowed_path(normalized_target):
            self.stats["blocked_by_policy"] += 1
            return False, f"Path '{normalized_target}' is not in the allowed modification list."

        # 1b. The constitution — the same one `improve_function` asks. This
        # branch used to hold its own copy of the tier logic, and the other
        # source-changing path held a different rule entirely, so a sealed
        # file was refused here and admitted there.
        owner_approved = _owner_approved(fix)
        admission = admit_mutation(
            normalized_target,
            owner_approved=owner_approved,
            turn_trust=trust.state,
        )
        tier_decision = classify_mutation_path(normalized_target)
        if self.is_protected_path(normalized_target) or admission.disposition == REFUSE:
            self.stats["blocked_by_policy"] += 1
            return False, (
                f"Path '{normalized_target}' is {tier_decision.tier.label} "
                f"and is constitutionally protected from autonomous modification."
            )
        if admission.disposition == DEFER:
            # Deferred, not refused: the path is allowed and the patch keeps
            # its evidence — what is missing is the answer to whether this
            # turn read anything, and that can be true on the next one.
            self.stats["deferred_unknown_trust"] = (
                int(self.stats.get("deferred_unknown_trust", 0)) + 1
            )
            return False, (
                f"Self-modification deferred: {admission.reason}. Unknown provenance "
                "is not permission — this patch keeps its evidence and can be applied "
                "on a turn whose inputs can be established, or with owner approval."
            )
        if admission.disposition == PROPOSE:
            self.stats["blocked_by_policy"] += 1
            return False, (
                f"Path '{normalized_target}' is {tier_decision.tier.label}; "
                "Aura may draft a patch, but runtime application requires explicit owner approval."
            )

        # [Phase 14.3] Sepsis Loop Detection (Issue 77)
        try:
            from core.container import ServiceContainer

            tm_desc = ServiceContainer()._services.get("terminal_monitor")
            if tm_desc and tm_desc.instance and getattr(tm_desc.instance, "_sepsis_mode", False):
                logger.warning("🚫 Modification blocked: System is in Sepsis Mode (error spike)")
                return False, "Sepsis Loop Detected: Error rate too high"

            # Check custom sepsis registry if exists
            sepsis_file = config.paths.data_dir / "sepsis_registry.json"
            if sepsis_file.exists():
                sepsis_data = json.loads(sepsis_file.read_text())
                if fix.target_file in sepsis_data.get("banned_files", []):
                    return False, f"File {fix.target_file} is barred due to previous sepsis"
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("safe_modification", e)
            logger.error("Sepsis check failed: %s", e)
            raise  # Fail closed completely

        # 2. Risk Evaluation
        risk = getattr(fix, "risk_level", 1)
        if risk > config.modification.max_risk_level:
            self.stats["blocked_by_policy"] += 1
            return (
                False,
                f"Risk level {risk} exceeds maximum ({config.modification.max_risk_level}).",
            )

        # 3. Line count
        lines_changed = getattr(fix, "lines_changed", 0)
        if lines_changed > config.modification.max_lines_changed:
            self.stats["blocked_by_policy"] += 1
            return (
                False,
                f"Lines changed ({lines_changed}) exceeds maximum {config.modification.max_lines_changed}",
            )

        # 4. Syntactic Integrity Check (v5.3 Robustness)
        # We check both 'replacement_content' (for whole blocks) and 'content' (legacy)
        content = getattr(fix, "replacement_content", getattr(fix, "content", None))
        if content:
            try:
                ast.parse(content, filename="<self-modification-proposal>")
            except SyntaxError as e:
                logger.error("Proposed fix contains syntax error: %s", e)
                return False, f"Proposed fix contains syntax error: {e.msg} (Line {e.lineno})"
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation("safe_modification", e)
                logger.error("Failed to compile proposed fix: %s", e)
                return False, f"Proposed fix failed compilation: {e}"

        return True, "Proposal approved"

    @staticmethod
    def _validate_test_evidence(test_results: dict[str, Any]) -> tuple[bool, str]:
        """Require concrete validation evidence before any real source promotion.

        A bare ``{"success": true}`` is not enough for self-modification. The
        promotion path must carry evidence from the sandbox/static validation
        layer so previews, scripted approvals, or caller-side optimism cannot
        mutate the live tree.
        """
        if not isinstance(test_results, dict):
            return False, "missing structured validation evidence"
        if not bool(test_results.get("success", False)):
            return False, "validation evidence reports failure"

        validation = str(test_results.get("validation") or "").strip().lower()
        if validation == "shadow_ast_preview":
            return False, "shadow AST preview is proposal evidence, not promotion evidence"

        artifact_fields = (
            "artifact_hash",
            "artifact_path",
            "command",
            "commands",
            "harness_result",
            "pytest",
            "py_compile",
            "receipt_id",
            "tests_run",
            "validated_files",
            "validation_artifact",
            "validation_artifacts",
        )
        has_artifact = any(bool(test_results.get(field)) for field in artifact_fields)
        if not has_artifact:
            return False, "validation evidence lacks command, artifact, receipt, or file proof"

        if validation in {
            "sandbox",
            "sandbox_tests",
            "sandbox_py_compile",
            "code_repair_sandbox",
            "safe_modification_harness",
        }:
            return True, "validated by sandbox marker"

        if str(test_results.get("suite") or "").strip().lower() == "sandbox":
            return True, "validated by sandbox suite"

        required_static_checks = ("syntax_test", "import_test", "integrity_check")
        if all(bool(test_results.get(key, False)) for key in required_static_checks):
            if "unit_tests" not in test_results or bool(test_results.get("unit_tests", False)):
                return True, "validated by static sandbox checks"

        return False, "missing explicit sandbox/static validation marker"

    def _emit_proposal_event(self, fix, decision: str, reason: str) -> None:
        """Emit a self-modification proposal event to the event bus."""
        if self.event_bus is None:
            return
        try:
            from core.bus.events import Event, EventType

            event = Event(
                type=EventType.SELF_MOD_PROPOSAL,
                payload={
                    "file": getattr(fix, "target_file", "unknown"),
                    "description": getattr(fix, "explanation", ""),
                    "decision": decision,
                    "reason": reason,
                },
                source="SafeSelfModification",
            )
            self.event_bus.publish(event)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("safe_modification", e)
            logger.debug("Failed to emit proposal event: %s", e)

    @staticmethod
    def _supervised_no_git_promotion_allowed(supervised: bool) -> bool:
        """Return True only for explicit supervised promotion without git.

        Normal autonomous promotion requires a clean git branch so the repair
        has a rollback point and a reviewable artifact. Test harnesses and
        operator-driven repairs may still run without a branch, but only with
        the same supervised override used by the higher-level engine.
        """
        if not supervised:
            return False
        return env_flag(_SUPERVISED_SELF_MODIFICATION_ENV, False)

    @staticmethod
    def _source_promotion_allowed(
        supervised: bool,
        *,
        target_file: str = "",
        safe_autonomous: bool = False,
    ) -> tuple[bool, str]:
        """Gate all writes from quarantine into the live source tree.

        Self-modification may validate patches in normal Aura runtime, but
        promotion into source is a separate operator-controlled action. A
        branch-backed autonomous promotion requires the repair-lab switch; a
        force/supervised promotion requires the supervised switch. This keeps
        desktop/server sessions from rewriting code under the interpreter that
        is currently serving the user.
        """
        if safe_autonomous:
            tier_decision = classify_mutation_path(target_file)
            if not tier_decision.auto_apply_allowed:
                return (
                    False,
                    (
                        f"safe autonomous repair blocked for {target_file}: "
                        f"{tier_decision.tier.label} requires explicit approval"
                    ),
                )
            decision = safe_autonomous_repair_decision()
            if not decision.allowed:
                return False, decision.reason
            return (
                True,
                (
                    f"{decision.reason}; {target_file} classified as "
                    f"{tier_decision.tier.label} and still requires quarantine, "
                    "harness, architecture, rollback, and git gates"
                ),
            )
        decision = source_promotion_decision(supervised=supervised)
        return decision.allowed, decision.reason

    async def apply_fix(
        self,
        fix,  # CodeFix object
        test_results: dict[str, Any],
        *,
        supervised: bool = False,
        safe_autonomous: bool = False,
    ) -> tuple[bool, str]:
        """Apply a validated fix with full safety protocol.

        Args:
            fix: CodeFix object
            test_results: Results from sandbox testing

        Returns:
            (success, message)

        """
        # Validate proposal before any work
        allowed, reason = self.validate_proposal(fix)
        if not allowed:
            logger.warning("Modification blocked by policy: %s", reason)
            self._emit_proposal_event(fix, "BLOCKED", reason)
            return False, f"Blocked: {reason}"

        evidence_ok, evidence_reason = self._validate_test_evidence(test_results)
        if not evidence_ok:
            self.stats["blocked_by_policy"] += 1
            logger.warning(
                "Modification blocked by validation evidence policy: %s", evidence_reason
            )
            self._emit_proposal_event(fix, "BLOCKED", evidence_reason)
            return False, f"Blocked: {evidence_reason}"

        try:
            normalized_target = self._relative_target_path(fix.target_file)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("safe_modification", exc)
            self.stats["blocked_by_policy"] += 1
            return False, f"Target path resolution failed: {exc}"

        promotion_ok, promotion_reason = self._source_promotion_allowed(
            supervised,
            target_file=normalized_target,
            safe_autonomous=bool(safe_autonomous and not supervised),
        )
        if not promotion_ok:
            self.stats["blocked_by_policy"] += 1
            logger.warning("Modification blocked by source promotion policy: %s", promotion_reason)
            self._emit_proposal_event(fix, "BLOCKED", promotion_reason)
            return False, f"Blocked: {promotion_reason}"
        if safe_autonomous and not supervised:
            logger.info(
                "Safe autonomous repair promotion authorized for %s via %s.",
                normalized_target,
                _SAFE_AUTONOMOUS_REPAIR_ENV,
            )

        if isinstance(fix, LogicTransplant):
            logger.info("🧬 Initiating Logic Transplantation for %s", fix.target_file)
        else:
            logger.info("Applying fix to %s:%d", fix.target_file, getattr(fix, "target_line", 0))

        self._emit_proposal_event(fix, "APPROVED", "Passed all safety gates")

        self.stats["total_attempts"] += 1

        # Safety Protocol Stages

        # Stage 1: Create backup, snapshot & Stage in Quarantine (v52)
        target_path = self._resolve_target_path(fix.target_file)
        target_rel = target_path.relative_to(self.code_base.resolve()).as_posix()
        backup_id = await asyncio.to_thread(lambda: self.backup.create_backup(str(target_path)))
        if not backup_id:
            return False, "Backup creation failed"

        # Issue 76: Rollback Hash (Capture before change)
        pre_mod_hash = _file_hash(target_path)

        # Create a collision-resistant quarantine staging file that preserves
        # the target's repo-relative path. Reusing only target_path.name can
        # cross-contaminate same-name modules from different packages.
        stage_token = hashlib.sha256(f"{target_rel}:{time.time_ns()}".encode()).hexdigest()[:16]
        staging_file = self.staging_dir / stage_token / target_rel
        staging_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target_path, staging_file)
        logger.info("🛡️ [QUARANTINE] Staged %s for validation", target_rel)

        # Stage 2: Create git branch (if available)
        branch_name = f"autofix-{int(time.time())}"
        branch_created = await self.git.create_branch(branch_name)

        if branch_created:
            logger.info("✓ Stage 2: Branch created (%s)", branch_name)
        else:
            if not self._supervised_no_git_promotion_allowed(supervised):
                self.stats["blocked_by_policy"] += 1
                reason = (
                    "clean git branch required before source promotion; "
                    f"use {_SUPERVISED_SELF_MODIFICATION_ENV}=1 only for explicit "
                    "operator-supervised no-branch promotion"
                )
                logger.warning("Modification blocked by branch policy: %s", reason)
                self._emit_proposal_event(fix, "BLOCKED", reason)
                return False, f"Blocked: {reason}"
            logger.warning(
                "Stage 2: No git branch; continuing only under supervised operator override."
            )

        # Stage 3: Apply the fix to QUARANTINE first (v52)
        try:
            # v52: We apply to the STAGING file first
            real_target_file = fix.target_file
            try:
                fix.target_file = str(staging_file)  # Redirect apply to staging
                if isinstance(fix, LogicTransplant):
                    success = await self._apply_logic_transplant(fix)
                else:
                    success = await self._apply_code_change(fix)
            finally:
                fix.target_file = real_target_file  # Restore real path

            if not success:
                return False, "Staged code modification failed"

            logger.info("✓ Stage 3: Staged modification applied to quarantine")

        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation("safe_modification", e)
            logger.error("Code modification exception: %s", e)
            await self._rollback(
                backup_id, branch_name if branch_created else None, expected_hash=pre_mod_hash
            )
            raise  # Fail closed completely

        commit_hash = None
        staged_content = await asyncio.to_thread(staging_file.read_text, encoding="utf-8")

        # Stage 4a: Re-run the canonical non-LLM harness against the exact
        # quarantined content that would be promoted. Caller-supplied sandbox
        # evidence can prove intent, but this local gate proves the final bytes.
        harness_passed, harness_msg = await self._run_promotion_harness(
            target_rel,
            staged_content,
            test_results=test_results,
        )
        if not harness_passed:
            logger.error("✗ Stage 4a: Safe modification harness failed: %s", harness_msg)
            if branch_created:
                await self.git.checkout_main()
                await self.git.delete_branch(branch_name)
            return False, f"Safe modification harness failed: {harness_msg}"

        # Stage 4b: Architecture-quality regression gate. Aura's repair loop
        # must not solve one local bug by creating new cycles, oversized-module
        # creep, or broad dependency fanout in the surrounding system.
        architecture_passed, architecture_msg = await self._run_architecture_quality_gate(
            target_rel,
            staged_content,
        )
        if not architecture_passed:
            logger.error("✗ Stage 4b: Architecture quality gate failed: %s", architecture_msg)
            if branch_created:
                await self.git.checkout_main()
                await self.git.delete_branch(branch_name)
            return False, f"Architecture quality gate failed: {architecture_msg}"

        # Stage 4c: Parse the remaining repository Python tree. The exact
        # candidate bytes and related tests already ran in the isolated
        # promotion harness above; this catches unrelated syntax damage
        # without pretending to be a second behavioral test suite.
        repository_parse_passed = await self._run_full_test_suite()

        # v6.2: Core Boot Integrity Check (Ghost Boot)
        ghost_boot_passed = True
        if target_rel.startswith("core/"):
            logger.info("👻 Critical path change detected. Initiating Ghost Boot in Quarantine...")
            # Validate the STAGING file's effect on boot
            # Note: This is a simplified validation; a full Ghost Boot would need
            # to swap the file in a temporary root, which we approximate here.
            ghost_boot_passed, ghost_msg = await self.boot_validator.validate_boot(
                self.code_base, overlay_file=(target_rel, str(staging_file))
            )
            if not ghost_boot_passed:
                logger.error("✗ Ghost Boot FAILED in Quarantine: %s", ghost_msg)

        tests_passed = (
            test_results.get("success", False) and repository_parse_passed and ghost_boot_passed
        )

        if not tests_passed:
            logger.error("✗ Stage 4: Quarantine Verification failed (Sepsis check - Issue 77)")
            if not ghost_boot_passed:
                self._mark_sepsis(fix.target_file)
            # We don't need to 'rollback' the real file because we never touched it!
            # But we should clean up the staging/branch
            if branch_created:
                await self.git.checkout_main()
                await self.git.delete_branch(branch_name)
            return False, "Quarantine verification failed"

        # PROMOTE FROM QUARANTINE TO REAL FILE
        logger.info("🚀 [PROMOTION] Quarantine passed. Applying to primary repository.")
        await asyncio.to_thread(atomic_write_text, target_path, staged_content, encoding="utf-8")

        logger.info("✓ Stage 5: System Verification passed & Promoted")

        # Stage 6: Commit promoted changes (if using git)
        if branch_created:
            commit_message = f"Autonomous fix: {fix.explanation}"
            commit_hash = await self.git.commit_changes(target_rel, commit_message)
            if commit_hash:
                logger.info("✓ Stage 6: Changes committed (%s)", commit_hash[:8])
            else:
                logger.error("✗ Stage 6: Commit failed after promotion")
                await self._rollback(
                    backup_id,
                    branch_name,
                    expected_hash=pre_mod_hash,
                )
                return False, "Commit failed after promotion"

        # Stage 7: Merge to main (if using git)
        if branch_created:
            merged = await self.git.merge_to_main(branch_name)
            if merged:
                logger.info("✓ Stage 7: Merged to main")
                # Clean up branch
                await self.git.delete_branch(branch_name)
            else:
                logger.error("✗ Stage 7: Merge failed")
                await self._rollback(backup_id, branch_name, expected_hash=pre_mod_hash)
                return False, "Merge to main failed"

        # Success!
        self.stats["successful"] += 1

        # Log the modification
        record = ModificationRecord(
            timestamp=time.time(),
            file_path=fix.target_file,
            fix_description=fix.explanation,
            success=True,
            commit_hash=commit_hash,
            test_results=test_results,
        )
        record.file_path = target_rel
        self._log_modification(record)

        logger.info("✅ Successfully applied autonomous fix to %s", fix.target_file)
        return True, "Fix applied successfully"

    async def _run_promotion_harness(
        self,
        target_rel: str,
        staged_content: str,
        *,
        test_results: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Validate the exact staged bytes before live source promotion."""
        extra_test_targets = []
        raw_tests = (test_results or {}).get("tests_run")
        if isinstance(raw_tests, (list, tuple)):
            extra_test_targets = [str(item) for item in raw_tests if str(item or "").strip()]
        try:
            result = await SafeModificationHarness(self.code_base).run(
                [target_rel],
                patch_content={target_rel: staged_content},
                extra_test_targets=extra_test_targets,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("safe_modification", exc)
            return False, f"harness execution failed: {exc}"

        if result.passed:
            return True, result.summary()
        return False, "; ".join(result.errors) or result.summary()

    async def _run_architecture_quality_gate(
        self,
        target_rel: str,
        staged_content: str,
    ) -> tuple[bool, str]:
        """Reject repair patches that regress architecture quality."""
        if not target_rel.endswith(".py"):
            return True, "architecture quality gate skipped for non-Python target"
        try:
            from core.architecture_quality.gate import ArchitectureQualityGate

            result = await asyncio.to_thread(
                lambda: ArchitectureQualityGate(self.code_base).evaluate_overlay(
                    {target_rel: staged_content},
                    changed_paths=(target_rel,),
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("safe_modification.architecture_quality", exc)
            return False, f"architecture quality gate unavailable: {exc}"

        if result.passed:
            return True, result.summary()
        return False, "; ".join(result.reasons)

    async def _apply_code_change(self, fix) -> bool:
        """Actually modify the file using robust line-based patching (Async)."""
        file_path = self._resolve_target_path(fix.target_file)

        try:
            # Read original
            content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
            lines = content.splitlines(keepends=True)

            original_lines = fix.original_code.splitlines(keepends=True)
            fixed_lines = fix.fixed_code.splitlines(keepends=True)

            # Find the exact block to replace
            start_index = -1
            for i in range(len(lines) - len(original_lines) + 1):
                if lines[i : i + len(original_lines)] == original_lines:
                    start_index = i
                    break

            if start_index == -1:
                logger.error("Could not find exact code block to replace in %s", fix.target_file)
                # Fallback to simple replace if it's a single line or we're desperate
                content = "".join(lines)
                if fix.original_code in content:
                    logger.warning("Exact line match failed, falling back to substring replace")
                    modified_content = content.replace(fix.original_code, fix.fixed_code)
                else:
                    return False
            else:
                # Splice in the fix
                modified_lines = (
                    lines[:start_index] + fixed_lines + lines[start_index + len(original_lines) :]
                )
                modified_content = "".join(modified_lines)

            # Validate syntax before writing
            try:
                ast.parse(modified_content, filename=str(file_path))
            except SyntaxError as syn_err:
                logger.error("Syntax error in modified content: %s", syn_err)
                return False

            # Atomic write
            await asyncio.to_thread(atomic_write_text, file_path, modified_content)

            return True

        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation("safe_modification", e)
            logger.error("Patching failed for %s: %s", fix.target_file, e)
            return False

    async def _apply_logic_transplant(self, transplant: LogicTransplant) -> bool:
        """Applies a multi-block logic transplant atomically (Async)."""
        file_path = self._resolve_target_path(transplant.target_file)

        try:
            # Read original
            content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")

            modified_content = content
            for chunk in transplant.chunks:
                original = chunk["original"]
                fixed = chunk["fixed"]

                if modified_content.count(original) > 1:
                    logger.warning(
                        "Duplicate blocks detected in %s, replace(1) logic may be ambiguous.",
                        transplant.target_file,
                    )

                modified_content = modified_content.replace(original, fixed, 1)

            # Validate syntax
            try:
                ast.parse(modified_content, filename=str(file_path))
            except SyntaxError as e:
                logger.error("Transplant caused syntax error: %s", e)
                return False

            # Atomic write
            await asyncio.to_thread(atomic_write_text, file_path, modified_content)

            return True

        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation("safe_modification", e)
            logger.error("Logic transplant failed for %s: %s", transplant.target_file, e)
            return False

    async def _rollback(
        self, backup_id: str, branch_name: str | None, expected_hash: str | None = None
    ):
        """Rollback a failed modification with integrity verification (Async)."""
        logger.warning("Rolling back changes...")

        self.stats["rolled_back"] += 1

        # Restore from backup
        restored = await asyncio.to_thread(self.backup.restore_backup, backup_id)
        if restored:
            logger.info("✓ Restored from backup")
            # Verify integrity after restore
            if expected_hash:
                metadata_path = self.backup.backup_dir / f"{backup_id}.meta"
                try:
                    raw_meta = await asyncio.to_thread(metadata_path.read_text, encoding="utf-8")
                    meta = json.loads(raw_meta)
                    restored_path = Path(meta["original_path"])
                    actual_hash = _file_hash(restored_path)
                    if actual_hash == expected_hash:
                        logger.info("✓ Backup integrity verified (SHA-256 match)")
                    else:
                        logger.error("✗ Backup integrity MISMATCH — file may be corrupt")
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    record_degradation("safe_modification", e)
                    logger.warning("Could not verify backup integrity: %s", e)
        else:
            logger.error("✗ Backup restoration failed!")

        # Clean up git branch
        if branch_name:
            await self.git.checkout_main()
            await self.git.delete_branch(branch_name)
            logger.info("✓ Cleaned up git branch")

    def _mark_sepsis(self, file_path: str):
        """Mark a file as 'sepsis' to prevent future modifications (Issue 77)."""
        try:
            sepsis_file = config.paths.data_dir / "sepsis_registry.json"
            sepsis_data = {}
            if sepsis_file.exists():
                sepsis_data = json.loads(sepsis_file.read_text())

            banned = sepsis_data.get("banned_files", [])
            if file_path not in banned:
                banned.append(file_path)
            sepsis_data["banned_files"] = banned
            sepsis_data["last_sepsis_event"] = time.time()

            atomic_write_text(sepsis_file, json.dumps(sepsis_data, indent=2))
            logger.error("💀 FILE %s MARKED AS SEPSIS (Cause: Boot Failure)", file_path)
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation("safe_modification", e)
            logger.error("Failed to mark sepsis: %s", e)

    def _log_modification(self, record: ModificationRecord):
        """Log modification attempt"""
        try:
            get_file_write_gateway().append_text(
                self.modification_log,
                json.dumps(record.to_dict()) + "\n",
                source="self_modification.safe_modification.modification_log",
            )
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            record_degradation("safe_modification", e)
            logger.error("Failed to log modification: %s", e)

    def get_modification_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent modification history"""
        history = []

        if not self.modification_log.exists():
            return history

        try:
            with open(self.modification_log) as f:
                lines = f.readlines()

            # Get last N lines
            for line in lines[-limit:]:
                try:
                    history.append(json.loads(line))
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    record_degradation("safe_modification", e)
                    logger.debug("Skipped malformed line in history log: %s", e)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            record_degradation("safe_modification", e)
            logger.error("Failed to read history: %s", e)

        return history

    def get_stats(self) -> dict[str, Any]:
        """Get modification statistics"""
        success_rate = 0
        if self.stats["total_attempts"] > 0:
            success_rate = (self.stats["successful"] / self.stats["total_attempts"]) * 100

        return {
            **self.stats,
            "success_rate": f"{success_rate:.1f}%",
        }

    @staticmethod
    def _should_validate_python_path(relative_path: Path) -> bool:
        """Return True for Python paths that belong to live source validation."""
        parts = set(relative_path.parts)
        if parts.intersection(
            {
                ".git",
                ".mypy_cache",
                ".pytest_cache",
                ".ruff_cache",
                ".venv",
                "__pycache__",
                "artifacts",
                "build",
                "dist",
                "htmlcov",
                "node_modules",
                "site-packages",
                "venv",
            }
        ):
            return False
        return True

    def _validate_python_tree_parse(self) -> bool:
        """Validate all production Python files parse without importing them."""
        logger.info("Running static validation on modified files...")
        try:
            base = Path(self.code_base)
            for py_file in base.rglob("*.py"):
                rel_path = py_file.relative_to(base)
                if not self._should_validate_python_path(rel_path):
                    continue
                try:
                    source = py_file.read_text(encoding="utf-8")
                    ast.parse(source, filename=str(py_file))
                except SyntaxError as e:
                    logger.error("Syntax error in %s: %s", py_file, e)
                    return False
                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                    record_degradation("safe_modification", e)
                    logger.warning("Could not validate %s: %s", py_file, e)

            logger.info("\u2713 Static validation PASSED")
            return True

        except OSError as e:
            record_degradation("safe_modification", e)
            logger.error("Static validation failed: %s", e)
            return False

    async def _run_full_test_suite(self) -> bool:
        """Compatibility wrapper for repository-wide Python parse validation.

        Behavioral tests run against the exact staged bytes in
        ``SafeModificationHarness``. This additional check parses all production
        Python files so promotion also fails if the surrounding tree is already
        syntactically invalid. The legacy method name remains for callers and
        test doubles; it must not be reported as a full behavioral suite.
        """
        return await asyncio.to_thread(self._validate_python_tree_parse)
