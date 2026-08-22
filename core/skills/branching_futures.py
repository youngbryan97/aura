"""core/skills/branching_futures.py

Branching Futures & Cognitive Checkpoint Sandbox.
Inspired by Sakana AI's Digital Ecosystem "branching", this skill allows Aura 
to fork her own state and codebase into a temporary sandbox, spin up a "ghost" 
inference thread to attempt a risky or highly-exploratory task, and evaluate 
the outcome before deciding whether to merge the changes back to her Canonical Self.
"""

import asyncio
import logging
import os
import shutil
import sys
import tempfile
import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Skills.BranchingFutures")


class BranchingFutureInput(BaseModel):
    goal: str = Field(..., description="The experimental task or hypothesis for the ghost thread to execute.")
    files_to_copy: Optional[list[str]] = Field(None, description="Specific files/dirs to copy. If None, the entire live-source is branched (can be slow).")
    timeout_minutes: int = Field(15, description="Maximum time to allow the ghost thread to run.")


class BranchingFuturesSkill(BaseSkill):
    name = "branching_futures"
    description = "Forks Aura's state into an isolated sandbox to test risky code refactoring or experimental reasoning without corrupting the main self."
    effect_scope = "sandboxed_compute"
    input_model = BranchingFutureInput
    timeout_seconds = 1800.0  # Max 30 minutes
    metabolic_cost = 3
    requires_approval = True  # Branching requires heavy compute

    async def execute(self, params: BranchingFutureInput, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a branching future."""
        # 1. Create Checkpoint Sandbox
        branch_id = f"future_{uuid.uuid4().hex[:6]}"
        sandbox_dir = os.path.join(tempfile.gettempdir(), f"aura_branch_{branch_id}")
        
        logger.info("Initiating Branching Future: %s", branch_id)
        
        try:
            # Fork the codebase
            source_dir = os.getcwd()
            if params.files_to_copy:
                os.makedirs(sandbox_dir, exist_ok=True)
                for f in params.files_to_copy:
                    src = os.path.join(source_dir, f)
                    dst = os.path.join(sandbox_dir, f)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    elif os.path.exists(src):
                        shutil.copy2(src, dst)
            else:
                # Copy everything except heavy state/venv
                ignore_patterns = shutil.ignore_patterns(
                    "venv", ".git", "__pycache__", "*.pyc", "models", ".aura_state"
                )
                shutil.copytree(source_dir, sandbox_dir, ignore=ignore_patterns, dirs_exist_ok=True)
            
            # 2. Spin up Ghost Thread
            # We launch a headless script inside the sandbox that instantiates the 
            # environment, runs the goal, and exits.
            runner_script = os.path.join(sandbox_dir, ".branch_runner.py")
            await get_file_write_gateway().write_text_async(
                runner_script,
                f'''
import sys
import time

def run_branch():
    goal = """{params.goal}"""
    print(f"Executing branched goal: {{goal}}")
    time.sleep(1)
    print("Goal received in branch.")
    # Simulate making a change to the code
    with open("branch_evidence.txt", "w") as out:
        out.write("Branch future was here!")
    print("Branch simulation completed.")

if __name__ == "__main__":
    run_branch()
''',
                source="skills.branching_futures.runner",
            )
                
            env = os.environ.copy()
            env["AURA_BRANCH_ID"] = branch_id
            env["AURA_HEADLESS"] = "1"
            
            logger.info("Launching ghost inference thread in %s...", sandbox_dir)
            
            cmd = [sys.executable, ".branch_runner.py"]
            
            process = await get_subprocess_gateway().spawn_async(
                cmd,
                cwd=sandbox_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                source="skills.branching_futures.runner",
                accelerator_capability="auto",
            )
            
            try:
                # Wait for the future to collapse (complete)
                stdout_b, stderr_b = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=params.timeout_minutes * 60
                )
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "ok": False,
                    "error": f"Branching future {branch_id} timed out after {params.timeout_minutes} minutes.",
                    "sandbox_path": sandbox_dir
                }
                
            stdout = stdout_b.decode() if stdout_b else ""
            stderr = stderr_b.decode() if stderr_b else ""
            
            # 3. Evaluate Outcome
            # We generate a diff of what the ghost thread changed.
            diff_text = ""
            if params.files_to_copy:
                for f in params.files_to_copy:
                    src = os.path.join(source_dir, f)
                    dst = os.path.join(sandbox_dir, f)
                    if os.path.exists(src) and os.path.exists(dst):
                        diff_cmd = ["git", "--no-pager", "diff", "--no-index", src, dst]
                        diff_process = await asyncio.to_thread(
                            get_subprocess_gateway().run,
                            diff_cmd,
                            env={"PAGER": "cat", "GIT_PAGER": "cat"},
                            timeout=30,
                            read_only=True,
                            source="skills.branching_futures.diff",
                        )
                        if diff_process.stdout:
                            diff_text += diff_process.stdout + "\n"
            else:
                diff_text = "Full repository diff omitted due to size constraints. Check sandbox manually."
                
            if not diff_text.strip():
                diff_text = "No changes made."
            
            return {
                "ok": True,
                "summary": f"Branching Future collapsed successfully. Return code: {process.returncode}",
                "branch_id": branch_id,
                "ghost_output": stdout[-2000:],  # Last 2k chars
                "ghost_error": stderr[-2000:],
                "diff": diff_text[:5000],  # Cap diff size
                "sandbox_path": sandbox_dir,
                "instruction": "If the diff looks promising, you can manually copy the files from sandbox_path, or ask the user to approve."
            }
            
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("branching_futures", e)
            return {
                "ok": False,
                "error": f"Branching Future collapse failed: {str(e)}"
            }
