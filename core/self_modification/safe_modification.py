"""Safe Self-Modification System
Version control integration with automatic rollback on failure.

v5.2: Added path allowlisting, risk gating, backup integrity verification,
      and event bus integration for modification proposals.
"""
import hashlib
import json
import logging
import os
import sys
import shutil
import subprocess
import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from .boot_validator import GhostBootValidator

logger = logging.getLogger("SelfModification.SafeModification")

from core.config import config


@dataclass
class ModificationRecord:
    """Record of a self-modification attempt"""

    timestamp: float
    file_path: str
    fix_description: str
    success: bool
    commit_hash: Optional[str] = None
    error: Optional[str] = None
    test_results: Optional[Dict[str, Any]] = None
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "file_path": self.file_path,
            "fix_description": self.fix_description,
            "success": self.success,
            "commit_hash": self.commit_hash,
            "error": self.error,
            "test_results": self.test_results
        }


@dataclass
class LogicTransplant:
    """Represents a multi-block architectural shift."""
    target_file: str
    explanation: str
    chunks: List[Dict[str, str]]  # List of {"original": "...", "fixed": "..."}
    risk_level: int = 5
    lines_changed: int = 0
    
    def to_dict(self):
        return {
            "target_file": self.target_file,
            "explanation": self.explanation,
            "chunks": self.chunks,
            "risk_level": self.risk_level,
            "lines_changed": self.lines_changed
        }


class GitIntegration:
    """Git version control integration for safe self-modification.
    """
    
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
            result = subprocess.run(
                ["git", "status"],
                cwd=self.repo_path,
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def _validate_path(file_path: str) -> str:
        """Issue 73/74: Validate and sanitize file paths (Link traversal & Unicode)."""
        import re
        # Block non-ASCII (Unicode path check - Issue 74)
        if not all(ord(c) < 128 for c in file_path):
            raise ValueError(f"Unicode characters not allowed in file paths: {file_path!r}")
            
        # Block shell metacharacters
        if re.search(r'[;&|`$(){}\[\]<>!\\\n\r]', file_path):
            raise ValueError(f"Path contains shell metacharacters: {file_path!r}")
            
        # Block path traversal (Issue 73)
        if '..' in file_path:
            raise ValueError(f"Path traversal detected: {file_path!r}")
            
        # Link traversal check (Issue 73)
        try:
            full_path = os.path.realpath(file_path)
            cwd_path = os.path.realpath(os.getcwd())
            if not full_path.startswith(cwd_path):
                raise ValueError(f"Path escaped project root via symlink: {file_path!r}")
        except Exception as e:
            if isinstance(e, ValueError): raise
            # If path doesn't exist yet, we can't realpath it fully, but we checked ..
            logger.debug("Path validation exception (likely non-existent): %s", e)

        # Block absolute paths outside the repo
        if file_path.startswith('/'):
            raise ValueError(f"Absolute paths not allowed: {file_path!r}")
            
        # Must be a Python file or known config
        if not file_path.endswith(('.py', '.toml', '.yaml', '.yml', '.json', '.cfg')):
            raise ValueError(f"Unsupported file type: {file_path!r}")
        return file_path

    @staticmethod
    def _validate_branch_name(branch_name: str) -> str:
        """C-04 FIX: Validate branch names to prevent injection."""
        import re
        if not re.match(r'^[a-zA-Z0-9_\-/\.]+$', branch_name):
            raise ValueError(f"Invalid branch name: {branch_name!r}")
        if len(branch_name) > 100:
            raise ValueError(f"Branch name too long: {len(branch_name)}")
        return branch_name
    
    async def _check_git_available(self) -> bool:
        """Check if git is installed and repo exists (Async)"""
        return await asyncio.to_thread(self._check_git_available_sync)

    async def is_worktree_dirty(self) -> bool:
        """Return True when the repository has any local changes."""
        if not self.git_available:
            return False
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["git", "status", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return bool(result.stdout.strip())
        except Exception as e:
            logger.debug("Dirty worktree check failed: %s", e)
            return True
    
    async def create_branch(self, branch_name: str) -> bool:
        """Create and checkout a new branch for testing fix (Async)."""
        if not self.git_available:
            logger.warning("Git not available, skipping branch creation")
            return False
        if await self.is_worktree_dirty():
            logger.info("Git worktree is dirty; skipping branch creation for safe autonomous fix flow.")
            return False
        
        try:
            branch_name = self._validate_branch_name(branch_name)
            # Create branch
            result = await asyncio.to_thread(
                subprocess.run,
                ["git", "checkout", "-b", branch_name],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info("Created branch: %s", branch_name)
                return True
            else:
                logger.error("Branch creation failed: %s", result.stderr)
                return False
                
        except Exception as e:
            logger.error("Branch creation exception: %s", e)
            return False
    
    async def commit_changes(self, file_path: str, message: str) -> Optional[str]:
        """Commit changes to current branch (Async)."""
        if not self.git_available:
            return None
        
        try:
            file_path = self._validate_path(file_path)
            # Stage file
            await asyncio.to_thread(
                subprocess.run,
                ["git", "add", file_path],
                cwd=self.repo_path,
                check=True,
                timeout=5
            )
            
            # Commit
            result = await asyncio.to_thread(
                subprocess.run,
                ["git", "commit", "-m", message],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logger.error("Commit failed: %s", result.stderr)
                return None
            
            # Get commit hash
            hash_result = await asyncio.to_thread(
                subprocess.run,
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            commit_hash = hash_result.stdout.strip()
            logger.info("Committed changes: %s", commit_hash[:8])
            return commit_hash
            
        except Exception as e:
            logger.error("Commit exception: %s", e)
            return None
    
    async def merge_to_main(self, branch_name: str) -> bool:
        """Merge branch into main after successful testing (Async)."""
        if not self.git_available:
            return False
        
        try:
            # Checkout main
            await asyncio.to_thread(
                subprocess.run,
                ["git", "checkout", "main"],
                cwd=self.repo_path,
                check=True,
                timeout=5
            )
            
            branch_name = self._validate_branch_name(branch_name)
            # Merge
            result = await asyncio.to_thread(
                subprocess.run,
                ["git", "merge", "--no-ff", branch_name, "-m", f"Auto-merge: {branch_name}"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info("Merged %s into main", branch_name)
                return True
            else:
                logger.error("Merge failed: %s", result.stderr)
                return False
                
        except Exception as e:
            logger.error("Merge exception: %s", e)
            return False
    
    async def delete_branch(self, branch_name: str) -> bool:
        """Delete a branch (Async)"""
        if not self.git_available:
            return False
        
        try:
            branch_name = self._validate_branch_name(branch_name)
            await asyncio.to_thread(
                subprocess.run,
                ["git", "branch", "-D", branch_name],
                cwd=self.repo_path,
                check=True,
                timeout=5
            )
            logger.info("Deleted branch: %s", branch_name)
            return True
        except Exception as e:
            logger.error("Branch deletion failed: %s", e)
            return False
    
    async def checkout_main(self) -> bool:
        """Return to main branch (Async)"""
        if not self.git_available:
            return False
        
        try:
            await asyncio.to_thread(
                subprocess.run,
                ["git", "checkout", "main"],
                cwd=self.repo_path,
                check=True,
                timeout=5
            )
            return True
        except Exception:
            return False
    
    async def get_current_branch(self) -> Optional[str]:
        """Get name of current branch (Async)"""
        if not self.git_available:
            return None
        
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip()
        except Exception:
            return None


class BackupSystem:
    """File-based backup system for when git is unavailable.
    """
    
    def __init__(self, backup_dir: Optional[str] = None):
        if backup_dir is None:
            from core.config import config
            self.backup_dir = config.paths.data_dir / "backups"
        else:
            self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info("BackupSystem initialized at %s", self.backup_dir)
    
    def create_backup(self, file_path: str) -> Optional[str]:
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
                "backup_id": backup_id
            }
            
            metadata_path = self.backup_dir / f"{backup_id}.meta"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            return backup_id
            
        except Exception as e:
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
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            original_path = Path(metadata["original_path"])
            
            # Restore file
            shutil.copy2(backup_path, original_path)
            logger.info("Restored backup %s to %s", backup_id, original_path)
            return True
            
        except Exception as e:
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
                except Exception as e:
                    logger.error("Cleanup failed for %s: %s", backup_file, e)


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
        modification_log: Optional[str] = None,
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
        return any(normalized.startswith(prefix) or f"/{prefix}" in normalized
                   for prefix in config.modification.allowed_paths)

    @staticmethod
    def is_protected_path(file_path: str) -> bool:
        """Check if a file path is inside a constitutionally protected area."""
        normalized = str(file_path).replace("\\", "/").lstrip("./")
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

    def validate_proposal(self, fix) -> Tuple[bool, str]:
        """Gate a modification proposal before it reaches apply_fix.

        Returns:
            (allowed, reason)
        """
        if not fix.target_file:
            return False, "No target file specified"

        try:
            normalized_target = self._relative_target_path(fix.target_file)
        except Exception as exc:
            self.stats["blocked_by_policy"] += 1
            return False, f"Target path resolution failed: {exc}"

        # 1. Path Allowlist Check
        if not self.is_allowed_path(normalized_target):
            self.stats["blocked_by_policy"] += 1
            return False, f"Path '{normalized_target}' is not in the allowed modification list."

        # 1b. Constitutional protected-path check
        if self.is_protected_path(normalized_target):
            self.stats["blocked_by_policy"] += 1
            return False, f"Path '{normalized_target}' is constitutionally protected from autonomous modification."

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
        except Exception as e:
            logger.debug("Sepsis check failed (non-blocking): %s", e)

        # 2. Risk Evaluation
        risk = getattr(fix, "risk_level", 1)
        if risk > config.modification.max_risk_level:
            self.stats["blocked_by_policy"] += 1
            return False, f"Risk level {risk} exceeds maximum ({config.modification.max_risk_level})."

        # 3. Line count
        lines_changed = getattr(fix, "lines_changed", 0)
        if lines_changed > config.modification.max_lines_changed:
            self.stats["blocked_by_policy"] += 1
            return False, f"Lines changed ({lines_changed}) exceeds maximum {config.modification.max_lines_changed}"

        # 4. Syntactic Integrity Check (v5.3 Robustness)
        # We check both 'replacement_content' (for whole blocks) and 'content' (legacy)
        content = getattr(fix, "replacement_content", getattr(fix, "content", None))
        if content:
            try:
                compile(content, "<self-modification-proposal>", "exec")
            except SyntaxError as e:
                logger.error("Proposed fix contains syntax error: %s", e)
                return False, f"Proposed fix contains syntax error: {e.msg} (Line {e.lineno})"
            except Exception as e:
                logger.error("Failed to compile proposed fix: %s", e)
                return False, f"Proposed fix failed compilation: {e}"

        return True, "Proposal approved"

    def _emit_proposal_event(self, fix, decision: str, reason: str) -> None:
        """Emit a self-modification proposal event to the event bus."""
        if self.event_bus is None:
            return
        try:
            from ..events import Event, EventType
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
        except Exception as e:
            logger.debug("Failed to emit proposal event: %s", e)

    @staticmethod
    def _file_hash(path: Path) -> str:
        """SHA-256 hash of a file for integrity verification."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    
    async def apply_fix(
        self,
        fix,  # CodeFix object
        test_results: Dict[str, Any]
    ) -> Tuple[bool, str]:
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
        pre_mod_hash = self._file_hash(target_path)
        
        # Create Quarantine Staging File
        staging_file = self.staging_dir / target_path.name
        shutil.copy2(target_path, staging_file)
        logger.info("🛡️ [QUARANTINE] Staged %s for validation", target_rel)
        
        # Stage 2: Create git branch (if available)
        branch_name = f"autofix-{int(time.time())}"
        branch_created = await self.git.create_branch(branch_name)
        
        if branch_created:
            logger.info("✓ Stage 2: Branch created (%s)", branch_name)
        else:
            logger.info("  Stage 2: No git branch (git unavailable)")
        
        # Stage 3: Apply the fix to QUARANTINE first (v52)
        try:
            # v52: We apply to the STAGING file first
            real_target_file = fix.target_file
            fix.target_file = str(staging_file) # Redirect apply to staging
            
            if isinstance(fix, LogicTransplant):
                success = await self._apply_logic_transplant(fix)
            else:
                success = await self._apply_code_change(fix)
                
            fix.target_file = real_target_file # Restore real path
            
            if not success:
                return False, "Staged code modification failed"
            
            logger.info("✓ Stage 3: Staged modification applied to quarantine")
            
        except Exception as e:
            logger.error("Code modification exception: %s", e)
            await self._rollback(backup_id, branch_name if branch_created else None, expected_hash=pre_mod_hash)
            return False, f"Modification exception: {e}"
        
        # Stage 4: Commit changes
        commit_hash = None
        if branch_created:
            commit_message = f"Autonomous fix: {fix.explanation}"
            commit_hash = await self.git.commit_changes(target_rel, commit_message)
            if commit_hash:
                logger.info("✓ Stage 4: Changes committed (%s)", commit_hash[:8])
            else:
                logger.warning("  Stage 4: Commit failed (continuing)")
        
        # Stage 5: Run comprehensive tests
        # For now, we trust the sandbox tests
        # Run full test suite
        full_suite_passed = await self._run_full_test_suite()
        
        # v6.2: Core Boot Integrity Check (Ghost Boot)
        ghost_boot_passed = True
        if target_rel.startswith("core/"):
            logger.info("👻 Critical path change detected. Initiating Ghost Boot in Quarantine...")
            # Validate the STAGING file's effect on boot
            # Note: This is a simplified validation; a full Ghost Boot would need
            # to swap the file in a temporary root, which we approximate here.
            ghost_boot_passed, ghost_msg = await self.boot_validator.validate_boot(self.code_base, overlay_file=(target_rel, str(staging_file)))
            if not ghost_boot_passed:
                logger.error("✗ Ghost Boot FAILED in Quarantine: %s", ghost_msg)

        tests_passed = test_results.get("success", False) and full_suite_passed and ghost_boot_passed
        
        if not tests_passed:
            logger.error("✗ Stage 5: Quarantine Verification failed (Sepsis check - Issue 77)")
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
        shutil.copy2(staging_file, target_path)
        
        logger.info("✓ Stage 5: System Verification passed & Promoted")
        
        # Stage 6: Merge to main (if using git)
        if branch_created:
            merged = await self.git.merge_to_main(branch_name)
            if merged:
                logger.info("✓ Stage 6: Merged to main")
                # Clean up branch
                await self.git.delete_branch(branch_name)
            else:
                logger.error("✗ Stage 6: Merge failed")
                await self._rollback(backup_id, branch_name)
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
            test_results=test_results
        )
        record.file_path = target_rel
        self._log_modification(record)
        
        logger.info("✅ Successfully applied autonomous fix to %s", fix.target_file)
        return True, "Fix applied successfully"
    
    async def _apply_code_change(self, fix) -> bool:
        """Actually modify the file using robust line-based patching (Async)."""
        file_path = self._resolve_target_path(fix.target_file)
        
        try:
            # Read original
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            original_lines = fix.original_code.splitlines(keepends=True)
            fixed_lines = fix.fixed_code.splitlines(keepends=True)
            
            # Find the exact block to replace
            start_index = -1
            for i in range(len(lines) - len(original_lines) + 1):
                if lines[i:i+len(original_lines)] == original_lines:
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
                modified_lines = lines[:start_index] + fixed_lines + lines[start_index+len(original_lines):]
                modified_content = "".join(modified_lines)
            
            # Validate syntax before writing
            import ast as _ast
            try:
                _ast.parse(modified_content)
            except SyntaxError as syn_err:
                logger.error("Syntax error in modified content: %s", syn_err)
                return False

            # Atomic write
            import tempfile as _tmpf
            fd, tmp_path = _tmpf.mkstemp(dir=os.path.dirname(file_path), suffix='.py.tmp')
            try:
                with os.fdopen(fd, 'w') as f:
                    f.write(modified_content)
                os.replace(tmp_path, file_path)
            except Exception:
                if os.path.exists(tmp_path): os.remove(tmp_path)
                raise
            
            return True
            
        except Exception as e:
            logger.error("Patching failed for %s: %s", fix.target_file, e)
            return False
            
    async def _apply_logic_transplant(self, transplant: LogicTransplant) -> bool:
        """Applies a multi-block logic transplant atomically (Async)."""
        file_path = self._resolve_target_path(transplant.target_file)
        
        try:
            # Read original
            with open(file_path, 'r') as f:
                content = f.read()
            
            modified_content = content
            for chunk in transplant.chunks:
                original = chunk["original"]
                fixed = chunk["fixed"]
                
                if modified_content.count(original) > 1:
                    logger.warning("Duplicate blocks detected in %s, replace(1) logic may be ambiguous.", transplant.target_file)
                
                modified_content = modified_content.replace(original, fixed, 1)
            
            # Validate syntax
            import ast
            try:
                ast.parse(modified_content)
            except SyntaxError as e:
                logger.error("Transplant caused syntax error: %s", e)
                return False
                
            # Atomic write
            import tempfile
            fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(file_path), suffix='.py.tmp')
            try:
                with os.fdopen(fd, 'w') as f:
                    f.write(modified_content)
                os.replace(tmp_path, file_path)
            except Exception:
                if os.path.exists(tmp_path): os.remove(tmp_path)
                raise
                
            return True
            
        except Exception as e:
            logger.error("Logic transplant failed for %s: %s", transplant.target_file, e)
            return False
    
    async def _rollback(self, backup_id: str, branch_name: Optional[str],
                  expected_hash: Optional[str] = None):
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
                    with open(metadata_path) as f:
                        meta = json.load(f)
                    restored_path = Path(meta["original_path"])
                    actual_hash = self._file_hash(restored_path)
                    if actual_hash == expected_hash:
                        logger.info("✓ Backup integrity verified (SHA-256 match)")
                    else:
                        logger.error("✗ Backup integrity MISMATCH — file may be corrupt")
                except Exception as e:
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
            
            sepsis_file.write_text(json.dumps(sepsis_data, indent=2))
            logger.error("💀 FILE %s MARKED AS SEPSIS (Cause: Boot Failure)", file_path)
        except Exception as e:
            logger.error("Failed to mark sepsis: %s", e)
    
    def _log_modification(self, record: ModificationRecord):
        """Log modification attempt"""
        try:
            with open(self.modification_log, 'a') as f:
                f.write(json.dumps(record.to_dict()) + '\n')
        except Exception as e:
            logger.error("Failed to log modification: %s", e)
    
    def get_modification_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent modification history"""
        history = []
        
        if not self.modification_log.exists():
            return history
        
        try:
            with open(self.modification_log, 'r') as f:
                lines = f.readlines()
            
            # Get last N lines
            for line in lines[-limit:]:
                try:
                    history.append(json.loads(line))
                except Exception as e:
                    logger.debug("Skipped malformed line in history log: %s", e)
        except Exception as e:
            logger.error("Failed to read history: %s", e)
        
        return history
    
    def get_stats(self) -> Dict[str, Any]:
        """Get modification statistics"""
        success_rate = 0
        if self.stats["total_attempts"] > 0:
            success_rate = (self.stats["successful"] / self.stats["total_attempts"]) * 100
        
        return {
            "success_rate": f"{success_rate:.1f}%"
        }

    async def _run_full_test_suite(self) -> bool:
        """Verify modified files using AST parsing (Async)."""
        """Verify modified files using AST parsing instead of executing tests.

        C-05 FIX: Replaced subprocess pytest execution with static AST
        validation. Running pytest in production is dangerous because:
        1. Tests may not exist in production deployments
        2. Test imports can execute production code with side effects
        3. The 60s timeout blocks the calling thread
        """
        import ast
        logger.info("Running static validation on modified files...")
        try:
            # Validate all Python files in the codebase parse correctly
            modified_files = []
            base = Path(self.code_base)
            for py_file in base.rglob("*.py"):
                # Skip test files, venv, and build dirs
                rel = str(py_file.relative_to(base))
                if any(skip in rel for skip in (".venv", "build", "dist", "__pycache__")):
                    continue
                try:
                    source = py_file.read_text(encoding="utf-8")
                    ast.parse(source, filename=str(py_file))
                except SyntaxError as e:
                    logger.error("Syntax error in %s: %s", py_file, e)
                    return False
                except Exception as e:
                    logger.warning("Could not validate %s: %s", py_file, e)

            logger.info("\u2713 Static validation PASSED")
            return True

        except Exception as e:
            logger.error("Static validation failed: %s", e)
            return False
