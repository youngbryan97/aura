"""GhostBoot: Shadow Boot-Path Validation
Verifies that Aura can still initialize after code modifications.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from core.runtime.errors import FallbackClassification, record_degradation

logger = logging.getLogger("SelfEvolution.GhostBoot")

_BOOT_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    subprocess.SubprocessError,
    TimeoutError,
)


def _record_boot_degradation(
    error: BaseException,
    *,
    action: str,
    stage: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    metadata = dict(extra or {})
    metadata["stage"] = stage
    try:
        record_degradation(
            "boot_validator",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=metadata,
        )
    except TypeError:
        record_degradation(
            "boot_validator",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
        )


def _safe_decode(data: bytes, *, max_chars: int = 20_000) -> str:
    text = data.decode(errors="replace") if data else ""
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def _sandbox_path(sandbox_path: Path, requested: str | Path) -> Path:
    root = sandbox_path.resolve()
    target = Path(requested)
    if not target.is_absolute():
        target = root / target
    resolved = target.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"ghost boot overlay target escapes sandbox: {requested}") from exc
    return resolved


class GhostBootValidator:
    """Validator that performs a 'trial boot' in a separate process."""

    def __init__(self, project_root: Path | None = None):
        from core.config import config

        self.root = project_root or config.paths.project_root

    async def validate_boot(
        self,
        sandbox_path: Path,
        timeout_s: int | float = 30,
        overlay_file: tuple[str, str] | None = None,
        **options: Any,
    ) -> tuple[bool, str]:
        """Attempts to 'boot' the system in the sandbox (Async)."""
        if "timeout" in options:
            timeout_s = options.pop("timeout")
        if options:
            unsupported = ", ".join(sorted(options))
            raise TypeError(f"unsupported ghost boot validation option(s): {unsupported}")

        logger.info("👻 Starting Ghost Boot validation in %s...", sandbox_path)

        # We use a specialized minimal boot script to speed up validation
        # and avoid side effects (like connecting to real APIs)
        boot_script = sandbox_path / ".ghost_boot.py"

        overlay_target: Path | None = None
        overlay_backup: bytes | None = None
        overlay_had_original = False

        try:
            await asyncio.to_thread(sandbox_path.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(self._create_minimal_boot_script, boot_script)

            if overlay_file:
                target_file, staging_file = overlay_file
                overlay_target = _sandbox_path(sandbox_path, target_file)
                staging_path = Path(staging_file)
                staging_exists = await asyncio.to_thread(staging_path.exists)
                if not staging_exists:
                    raise FileNotFoundError(f"ghost boot staging file missing: {staging_path}")
                await asyncio.to_thread(overlay_target.parent.mkdir, parents=True, exist_ok=True)
                overlay_had_original = await asyncio.to_thread(overlay_target.exists)
                if overlay_had_original:
                    overlay_backup = await asyncio.to_thread(overlay_target.read_bytes)
                await asyncio.to_thread(shutil.copy2, staging_path, overlay_target)

            # Run the boot script in a subprocess
            env = os.environ.copy()
            env["PYTHONPATH"] = str(sandbox_path)
            env["AURA_GHOST_BOOT"] = "1"  # Flag to skip heavy systems

            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(boot_script),
                cwd=str(sandbox_path),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_s
                )
                output = _safe_decode(stdout_data)
                error_output = _safe_decode(stderr_data)
            except TimeoutError:
                process.kill()
                stdout_data, stderr_data = await process.communicate()
                output = _safe_decode(stdout_data)
                error_output = _safe_decode(stderr_data)
                error_output += "\n[Timeout reached without heartbeat]"

            success = "GHOST_HEARTBEAT_STABLE" in output

            if success:
                logger.info("✅ Ghost Boot SUCCESS: System reached stable state.")
                return True, "Stable boot reached"
            else:
                error_msg = error_output or "Timeout reached without heartbeat"
                logger.error("❌ Ghost Boot FAILED: %s", error_msg)
                return False, f"Boot failure: {error_msg}"

        except _BOOT_RECOVERABLE_ERRORS as e:
            _record_boot_degradation(
                e,
                action="failed ghost boot validation without promoting candidate changes",
                stage="validate_boot",
                severity="degraded",
                extra={"sandbox_path": str(sandbox_path)},
            )
            logger.error("Ghost Boot execution error: %s", e)
            return False, str(e)
        finally:
            if overlay_target is not None:
                try:
                    if overlay_had_original and overlay_backup is not None:
                        await asyncio.to_thread(overlay_target.write_bytes, overlay_backup)
                    elif await asyncio.to_thread(overlay_target.exists):
                        await asyncio.to_thread(overlay_target.unlink)
                except _BOOT_RECOVERABLE_ERRORS as e:
                    _record_boot_degradation(
                        e,
                        action="left ghost boot overlay for operator cleanup after restore failed",
                        stage="cleanup.overlay_restore",
                        extra={"overlay_target": str(overlay_target)},
                    )
                    logger.debug("Failed to restore ghost boot overlay: %s", e)
            if await asyncio.to_thread(boot_script.exists):
                try:
                    await asyncio.to_thread(boot_script.unlink)
                except _BOOT_RECOVERABLE_ERRORS as e:
                    _record_boot_degradation(
                        e,
                        action="left ghost boot script for operator cleanup after unlink failed",
                        stage="cleanup.boot_script",
                        extra={"boot_script": str(boot_script)},
                    )
                    logger.debug("Failed to unlink ghost boot script: %s", e)

    def _create_minimal_boot_script(self, script_path: Path):
        """Creates a script that initializes key Aura subsystems for verification."""
        content = """
import os
import sys
import logging
import time
import types

# Create a dummy logger to avoid configuration issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GhostBoot")

# Disable all heavy systems immediately
os.environ["AURA_GHOST_BOOT"] = "1"
os.environ["AURA_INTERNAL_ONLY"] = "1"

# Stub out blocking components before importing (lightweight, no unittest.mock)
_stub = types.ModuleType("core.managers.health_monitor")
class _GhostHealthMonitor:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.started_at = time.time()
    def get_status(self):
        return {"status": "ghost", "uptime": time.time() - self.started_at}
    def is_healthy(self):
        return True
_stub.HealthMonitor = _GhostHealthMonitor
sys.modules["core.managers.health_monitor"] = _stub

try:
    logger.info("Ghost: Initializing core architecture...")
    from core.config import config
    from core.orchestrator import RobustOrchestrator
    from core.state.aura_state import AuraState
    
    # Minimal init
    from core.state.aura_state import AuraState
    orch = RobustOrchestrator()
    orch.state = AuraState() # Ensure fresh state
    
    logger.info("Ghost: Orchestrator initialized. Testing state derivation...")
    
    # Test a simple state derivation to ensure lineage/versioning is intact
    test_state = orch.state.derive("ghost_test", origin="GhostBoot")
    if test_state.parent_state_id != orch.state.state_id:
        raise RuntimeError("State lineage failed in Ghost Boot")
    if test_state.version != orch.state.version + 1:
        raise RuntimeError("State derivation failed in Ghost Boot")

    logger.info("Ghost: Orchestrator Status -> %s", orch.status)
    
    # We reached the end of validation without crashing
    print("GHOST_HEARTBEAT_STABLE")
    sys.exit(0)
except (ImportError, AttributeError, RuntimeError) as e:
    import traceback
    logger.error("Ghost: BOOT CRASH: %s", e)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(content)
