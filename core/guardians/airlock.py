from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import shutil
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import config
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.tasks.managed_command import ManagedCommandResult

logger = logging.getLogger("Aura.Airlock")
_PIPE = -1
_AIRLOCK_RECOVERABLE_ERRORS = (
    FileNotFoundError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
CommandRunner = Callable[[tuple[str, ...], Path, float], Awaitable[ManagedCommandResult]]


class AirlockCommandError(RuntimeError):
    def __init__(self, action: str, result: ManagedCommandResult):
        self.action = action
        self.result = result
        detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
        super().__init__(f"{action} failed: {detail[:300]}")


@dataclass(frozen=True)
class AirlockGauntletReport:
    passed: bool
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class AirlockProtocol:
    """
    The Meta-Controller Airlock.
    Forces all self-modifications (mutations) to occur on an ephemeral git branch, 
    run the test suite, and compile a benchmark report before merging.
    """
    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        sandbox_dir: Path | None = None,
        command_runner: CommandRunner | None = None,
        test_timeout_s: float = 600.0,
    ):
        self.repo_root = Path(repo_root or config.paths.project_root).resolve()
        self.sandbox_dir = Path(sandbox_dir or (Path.home() / ".aura" / "tmp" / "airlock_sandbox")).resolve()
        self._command_runner = command_runner or self._default_command_runner
        self.test_timeout_s = test_timeout_s
        
    async def process_mutation(self, hypothesis_id: str, diff_patch: str, description: str) -> dict[str, Any]:
        """
        Executes the Airlock Gauntlet on a proposed code mutation.
        """
        logger.warning("🔒 AIRLOCK PROTOCOL ENGAGED for Mutation %s", hypothesis_id)
        
        branch_name = f"mutation/{self._safe_branch_suffix(hypothesis_id)}"
        
        try:
            with tempfile.TemporaryDirectory(prefix="aura_airlock_patch_") as patch_tmp:
                patch_file = Path(patch_tmp) / f"{self._safe_branch_suffix(hypothesis_id)}.patch"
                atomic_write_text(patch_file, diff_patch, encoding="utf-8")

                # 1. Setup Sandbox
                if self.sandbox_dir.exists():
                    shutil.rmtree(self.sandbox_dir)
                await self._prepare_worktree()

                # 2. Git Operations (Branch, Patch) inside ephemeral worktree
                logger.info("  [1/4] Preparing ephemeral Git worktree...")
                await self._run_git(("checkout", "-B", branch_name), cwd=self.sandbox_dir, action="checkout branch")

                # Apply Patch
                logger.info("  [2/4] Applying mutation diff...")
                await self._run_git(("apply", "--check", str(patch_file)), cwd=self.sandbox_dir, action="patch check")
                await self._run_git(("apply", str(patch_file)), cwd=self.sandbox_dir, action="patch apply")

                # Commit only tracked project changes; the patch artifact lives outside the worktree.
                await self._run_git(("add", "-A"), cwd=self.sandbox_dir, action="git add")
                await self._run_git(
                    ("commit", "-m", f"[AUTONOMOUS_MUTATION] {description}", "--no-gpg-sign"),
                    cwd=self.sandbox_dir,
                    action="git commit",
                )
                
                # 3. The Gauntlet (Testing)
                logger.info("  [3/4] Running the Gauntlet (Pytest Suite)...")
                gauntlet = await self._run_tests()
                
                if not gauntlet.passed:
                    logger.error("❌ Airlock Gauntlet FAILED. Mutation rejected.")
                    await self._cleanup_worktree()
                    return {
                        "success": False,
                        "reason": "Test suite failed.",
                        "stdout": gauntlet.stdout,
                        "stderr": gauntlet.stderr,
                        "timed_out": gauntlet.timed_out,
                    }

                # 4. Success / Hand-off
                logger.info("  [4/4] Gauntlet passed. Awaiting human merge review.")
                await self._cleanup_worktree()
            
                return {
                    "success": True,
                    "branch": branch_name,
                    "message": "Mutation passed tests and was committed to ephemeral branch. Ready for merge.",
                    "stdout": gauntlet.stdout,
                    "stderr": gauntlet.stderr,
                }
            
        except AirlockCommandError as e:
            logger.error("Airlock command error: %s", e)
            await self._recover_state()
            return {"success": False, "reason": str(e), "stdout": e.result.stdout, "stderr": e.result.stderr}
        except _AIRLOCK_RECOVERABLE_ERRORS as e:
            record_degradation('airlock', e)
            logger.error("Airlock Exception: %s", e)
            await self._recover_state()
            return {"success": False, "reason": str(e)}

    async def _run_git(
        self,
        cmd_args: tuple[str, ...],
        *,
        cwd: Path | None = None,
        action: str = "git command",
    ) -> str:
        result = await self._command_runner(("git", *cmd_args), cwd or self.repo_root, 60.0)
        if not result.ok:
            raise AirlockCommandError(action, result)
        return result.stdout

    async def _prepare_worktree(self) -> None:
        await self._run_git(("worktree", "prune"), action="worktree prune")
        await self._run_git(("worktree", "add", "--detach", str(self.sandbox_dir), "HEAD"), action="worktree add")

    async def _cleanup_worktree(self) -> None:
        try:
            await self._run_git(("worktree", "remove", "--force", str(self.sandbox_dir)), action="worktree cleanup")
        except (AirlockCommandError, *_AIRLOCK_RECOVERABLE_ERRORS) as e:
            record_degradation('airlock', e)
            logger.warning("Airlock worktree cleanup failed: %s", e)
        
    async def _run_tests(self) -> AirlockGauntletReport:
        """Run pytest to validate the new code."""
        result = await self._command_runner(
            (sys.executable, "-m", "pytest", "-q"),
            self.sandbox_dir,
            self.test_timeout_s,
        )
        return AirlockGauntletReport(
            passed=result.ok,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=result.timed_out,
        )

    async def _recover_state(self):
        """Emergency recovery that avoids mutating the live repository state."""
        logger.warning("Attempting Airlock State Recovery...")
        try:
            await self._cleanup_worktree()
        except _AIRLOCK_RECOVERABLE_ERRORS as e:
            record_degradation('airlock', e)
            logger.critical("FATAL: Airlock failed to recover base state! %s", e)

    @staticmethod
    async def _default_command_runner(
        command: tuple[str, ...],
        cwd: Path,
        timeout_s: float,
    ) -> ManagedCommandResult:
        started = time.perf_counter()
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            stdout=_PIPE,
            stderr=_PIPE,
        )
        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        except TimeoutError:
            timed_out = True
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            stdout_bytes, stderr_bytes = await process.communicate()
        return ManagedCommandResult(
            command,
            process.returncode,
            stdout_bytes.decode("utf-8", errors="replace"),
            stderr_bytes.decode("utf-8", errors="replace"),
            time.perf_counter() - started,
            timed_out=timed_out,
        )

    @staticmethod
    def _safe_branch_suffix(hypothesis_id: str) -> str:
        parts: list[str] = []
        previous_separator = False
        for ch in hypothesis_id.strip():
            if ch.isalnum() or ch in "_-":
                parts.append(ch)
                previous_separator = False
            elif not previous_separator:
                parts.append("-")
                previous_separator = True

        normalized = "".join(parts).strip("-_")[:80]
        if normalized:
            return normalized
        return hashlib.sha256(hypothesis_id.encode("utf-8", errors="replace")).hexdigest()[:16]
