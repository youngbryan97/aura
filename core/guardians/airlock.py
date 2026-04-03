import asyncio
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from core.config import config

logger = logging.getLogger("Aura.Airlock")

class AirlockProtocol:
    """
    The Meta-Controller Airlock.
    Forces all self-modifications (mutations) to occur on an ephemeral git branch, 
    run the test suite, and compile a benchmark report before merging.
    """
    def __init__(self):
        self.repo_root = config.paths.project_root
        self.sandbox_dir = Path.home() / ".aura" / "tmp" / "airlock_sandbox"
        
    async def process_mutation(self, hypothesis_id: str, diff_patch: str, description: str) -> Dict[str, Any]:
        """
        Executes the Airlock Gauntlet on a proposed code mutation.
        """
        logger.warning(f"🔒 AIRLOCK PROTOCOL ENGAGED for Mutation {hypothesis_id}")
        
        branch_name = f"mutation/{hypothesis_id}"
        patch_file = self.sandbox_dir / f"{hypothesis_id}.patch"
        
        try:
            # 1. Setup Sandbox
            if self.sandbox_dir.exists():
                shutil.rmtree(self.sandbox_dir)
            self._prepare_worktree()
            with open(patch_file, "w") as f:
                f.write(diff_patch)
                
            # 2. Git Operations (Branch, Patch) inside ephemeral worktree
            logger.info("  [1/4] Preparing ephemeral Git worktree...")
            try:
                self._run_git(["checkout", "-b", branch_name], cwd=self.sandbox_dir)
            except subprocess.CalledProcessError:
                # Branch might exist if recovering from crash
                self._run_git(["branch", "-D", branch_name])
                self._run_git(["checkout", "-b", branch_name], cwd=self.sandbox_dir)

            # Apply Patch
            logger.info("  [2/4] Applying mutation diff...")
            self._run_git(["apply", str(patch_file)], cwd=self.sandbox_dir)
            
            # Commit
            self._run_git(["add", "."], cwd=self.sandbox_dir)
            self._run_git(["commit", "-m", f"[AUTONOMOUS_MUTATION] {description}"], cwd=self.sandbox_dir)
            
            # 3. The Gauntlet (Testing)
            logger.info("  [3/4] Running the Gauntlet (Pytest Suite)...")
            test_passed = self._run_tests()
            
            if not test_passed:
                logger.error("❌ Airlock Gauntlet FAILED. Mutation rejected.")
                self._cleanup_worktree()
                return {"success": False, "reason": "Test suite failed."}
                
            # 4. Success / Hand-off
            logger.info("  [4/4] Gauntlet passed. Awaiting human merge review.")
            self._cleanup_worktree()
            
            return {
                "success": True, 
                "branch": branch_name, 
                "message": "Mutation passed tests and was committed to ephemeral branch. Ready for merge."
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Airlock Git Exception: {e}")
            self._recover_state()
            return {"success": False, "reason": f"Git/Patch Error: {e.output}"}
        except Exception as e:
            logger.error(f"Airlock Exception: {e}")
            self._recover_state()
            return {"success": False, "reason": str(e)}

    def _run_git(self, cmd_args: list, cwd: Optional[Path] = None) -> str:
        cmd = ["git"] + cmd_args
        result = subprocess.run(
            cmd, 
            cwd=cwd or self.repo_root,
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout

    def _prepare_worktree(self) -> None:
        self._run_git(["worktree", "prune"])
        self._run_git(["worktree", "add", "--detach", str(self.sandbox_dir), "main"])

    def _cleanup_worktree(self) -> None:
        try:
            self._run_git(["worktree", "remove", "--force", str(self.sandbox_dir)])
        except Exception as e:
            logger.warning("Airlock worktree cleanup failed: %s", e)
        
    def _run_tests(self) -> bool:
        """Run pytest to validate the new code."""
        try:
            result = subprocess.run(
                ["pytest"], 
                cwd=self.sandbox_dir, 
                capture_output=True, 
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            # If pytest isn't installed in the path exactly like this, fail gracefully or assume pass if not enforced
            logger.warning("pytest executable not found in PATH for Airlock. Faling test automatically.")
            return False

    def _recover_state(self):
        """Emergency recovery that avoids mutating the live repository state."""
        logger.warning("Attempting Airlock State Recovery...")
        try:
            self._cleanup_worktree()
        except Exception as e:
            logger.critical(f"FATAL: Airlock failed to recover base state! {e}")
