import ast
import asyncio
import logging
from pathlib import Path

logger = logging.getLogger("Kernel.Mutate")

async def _run_tests_async() -> tuple[bool, str]:
    """Return (ok, msg)."""
    try:
        # Runs pytest in the current environment asynchronously.
        process = await asyncio.create_subprocess_exec(
            "pytest", "-q",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            return True, stdout.decode().strip()
        return False, (stderr.decode().strip() or stdout.decode().strip())
    except Exception as e:
        return False, f"pytest failed to run: {e}"

async def apply_mutation(target_path: str, new_code: str) -> bool:
    """Safely apply a mutation using Git for atomic rollback (Async)."""
    path = Path(target_path)

    if not path.exists():
        logger.error("Target file does not exist: %s", path)
        return False

    # 1. Syntax check
    try:
        ast.parse(new_code)
    except SyntaxError as e:
        logger.error("Mutation rejected: SyntaxError: %s", e)
        return False

    logger.info("Initiating Mutation on %s", path.name)
    
    # 2. Git Checkpoint
    try:
        proc1 = await asyncio.create_subprocess_exec("git", "add", ".", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc1.communicate()
        
        proc2 = await asyncio.create_subprocess_exec(
            "git", "commit", "-m", f"Checkpoint Pre-Mutation: {path.name}",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await proc2.communicate()
    except Exception as e:
        logger.error("Failed to create pre-mutation checkpoint: %s", e)
        return False

    try:
        # 3. Write candidate
        path.write_text(new_code, encoding="utf-8")
        logger.info("Wrote candidate mutation to %s", path)

        # 4. Tests
        logger.info("Running Trial (pytest)...")
        ok, msg = await _run_tests_async()
        
        if not ok:
            logger.error("Trial Failed: %s", msg)
            # 5. Rollback via Git
            logger.warning("Rolling back via Git Reset...")
            proc_reset = await asyncio.create_subprocess_exec("git", "reset", "--hard", "HEAD", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await proc_reset.communicate()
            return False

        # 6. Commit mutation (Finalize)
        logger.info("Trial Passed. Committing changes.")
        proc_add = await asyncio.create_subprocess_exec("git", "add", str(path), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc_add.communicate()
        
        proc_commit = await asyncio.create_subprocess_exec(
            "git", "commit", "-m", f"Mutation Applied: {path.name}",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await proc_commit.communicate()
        return True

    except Exception as e:
        logger.exception("Critical error during mutation: %s", e)
        # Emergency Rollback
        proc_emerg = await asyncio.create_subprocess_exec("git", "reset", "--hard", "HEAD", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc_emerg.communicate()
        return False
