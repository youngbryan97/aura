"""GhostBoot: Shadow Boot-Path Validation
Verifies that Aura can still initialize after code modifications.
"""

import asyncio
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from core.runtime.errors import FallbackClassification, record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.subprocess_gateway import get_subprocess_gateway

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

        logger.info("Starting Ghost Boot validation in %s...", sandbox_path)

        validation_root = sandbox_path
        isolated_workspace: tempfile.TemporaryDirectory[str] | None = None
        boot_script = validation_root / ".ghost_boot.py"

        try:
            if overlay_file:
                target_file, staging_file = overlay_file
                requested_target = _sandbox_path(sandbox_path, target_file)
                staging_path = Path(staging_file)
                staging_exists = await asyncio.to_thread(staging_path.exists)
                if not staging_exists:
                    raise FileNotFoundError(f"ghost boot staging file missing: {staging_path}")
                sandbox_root = await asyncio.to_thread(sandbox_path.resolve)
                relative_target = requested_target.relative_to(sandbox_root).as_posix()
                staged_content = await asyncio.to_thread(staging_path.read_text, encoding="utf-8")
                isolated_workspace = tempfile.TemporaryDirectory(
                    prefix="aura-ghost-boot-candidate-"
                )
                validation_root = Path(isolated_workspace.name)
                from core.self_modification.safe_modification_harness import (
                    SafeModificationHarness,
                )

                harness = SafeModificationHarness(sandbox_path)
                await asyncio.to_thread(
                    harness.prepare_candidate_workspace,
                    validation_root,
                    {relative_target: staged_content},
                )

            await asyncio.to_thread(validation_root.mkdir, parents=True, exist_ok=True)
            boot_script = validation_root / ".ghost_boot.py"
            await asyncio.to_thread(self._create_minimal_boot_script, boot_script)

            # Run the boot script in a subprocess
            env = os.environ.copy()
            env["PYTHONPATH"] = str(validation_root)
            env["AURA_GHOST_BOOT"] = "1"  # Flag to skip heavy systems

            process = await get_subprocess_gateway().spawn_async(
                [sys.executable, str(boot_script)],
                cwd=str(validation_root),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                source="self_modification:ghost_boot_validator",
                accelerator_capability="none",
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
                logger.info("Ghost Boot SUCCESS: System reached stable state.")
                return True, "Stable boot reached"
            else:
                error_msg = error_output or "Timeout reached without heartbeat"
                logger.error("Ghost Boot FAILED: %s", error_msg)
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
            if isolated_workspace is not None:
                await asyncio.to_thread(isolated_workspace.cleanup)

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
        get_file_write_gateway().write_text(
            script_path,
            content,
            source="self_modification.boot_validator.script",
        )
