import ast
import asyncio
import logging
import subprocess
from pathlib import Path

from core.runtime.atomic_writer import async_atomic_write_text, atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Kernel.Mutate")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_mutation_target(target_path: str) -> Path | None:
    """Return a repo-contained mutation target, or None if the path is unsafe."""
    try:
        path = Path(target_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        resolved = path.resolve()
        resolved.relative_to(PROJECT_ROOT)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("Mutation rejected: unsafe target path %r: %s", target_path, exc)
        return None

    if ".git" in resolved.relative_to(PROJECT_ROOT).parts:
        logger.error("Mutation rejected: refusing to edit git metadata: %s", resolved)
        return None
    return resolved


async def _run_tests_async() -> tuple[bool, str]:
    """Return (ok, msg)."""
    try:
        # Runs pytest in the current environment asynchronously.
        process = await get_subprocess_gateway().spawn_async(
            ["pytest", "-q"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            source="self_modification:mutate.pytest",
            accelerator_capability="none",
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            return True, stdout.decode().strip()
        return False, (stderr.decode().strip() or stdout.decode().strip())
    except (subprocess.SubprocessError, OSError) as e:
        record_degradation('mutate', e)
        return False, f"pytest failed to run: {e}"

async def apply_mutation(target_path: str, new_code: str) -> bool:
    """Safely apply a repo-contained mutation and roll it back on failure."""
    path = _resolve_mutation_target(target_path)
    if path is None:
        return False

    if not path.exists():
        logger.error("Target file does not exist: %s", path)
        return False
    if not path.is_file():
        logger.error("Mutation rejected: target is not a file: %s", path)
        return False

    # 1. Syntax check
    try:
        ast.parse(new_code)
    except SyntaxError as e:
        logger.error("Mutation rejected: SyntaxError: %s", e)
        return False

    logger.info("Initiating Mutation on %s", path.name)
    original_text = path.read_text(encoding="utf-8")

    try:
        # 2. Write candidate
        await async_atomic_write_text(path, new_code, encoding="utf-8")
        logger.info("Wrote candidate mutation to %s", path)

        # 3. Tests
        logger.info("Running Trial (pytest)...")
        ok, msg = await _run_tests_async()
        
        if not ok:
            logger.error("Trial Failed: %s", msg)
            logger.warning("Rolling back mutation by restoring original file contents.")
            await async_atomic_write_text(path, original_text, encoding="utf-8")
            return False

        # 4. Commit mutation (Finalize). Only stage the target file.
        logger.info("Trial Passed. Committing changes.")
        rel_path = str(path.relative_to(PROJECT_ROOT))
        proc_add = await get_subprocess_gateway().spawn_async(
            ["git", "add", rel_path],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            source="self_modification:mutate.git_add",
            accelerator_capability="none",
        )
        await proc_add.communicate()
        
        proc_commit = await get_subprocess_gateway().spawn_async(
            ["git", "commit", "-m", f"Mutation Applied: {path.name}"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            source="self_modification:mutate.git_commit",
            accelerator_capability="none",
        )
        await proc_commit.communicate()
        return True

    except (subprocess.SubprocessError, OSError) as e:
        record_degradation('mutate', e)
        logger.exception("Critical error during mutation: %s", e)
        await async_atomic_write_text(path, original_text, encoding="utf-8")
        return False
