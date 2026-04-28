"""GhostBoot: Shadow Boot-Path Validation
Verifies that Aura can still initialize after code modifications.
"""
from core.runtime.errors import record_degradation
import os
import sys
import subprocess
import time
import asyncio
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("SelfEvolution.GhostBoot")

class GhostBootValidator:
    """Validator that performs a 'trial boot' in a separate process."""

    def __init__(self, project_root: Optional[Path] = None):
        from core.config import config
        self.root = project_root or config.paths.project_root

    async def validate_boot(
        self,
        sandbox_path: Path,
        timeout: int = 30,
        overlay_file: Optional[Tuple[str, str]] = None,
    ) -> Tuple[bool, str]:
        """Attempts to 'boot' the system in the sandbox (Async)."""
        logger.info("👻 Starting Ghost Boot validation in %s...", sandbox_path)
        
        # We use a specialized minimal boot script to speed up validation
        # and avoid side effects (like connecting to real APIs)
        boot_script = sandbox_path / ".ghost_boot.py"
        self._create_minimal_boot_script(boot_script)

        overlay_target: Optional[Path] = None
        overlay_backup: Optional[bytes] = None
        overlay_had_original = False

        try:
            if overlay_file:
                target_file, staging_file = overlay_file
                overlay_target = Path(target_file)
                if not overlay_target.is_absolute():
                    overlay_target = sandbox_path / overlay_target
                staging_path = Path(staging_file)
                overlay_target.parent.mkdir(parents=True, exist_ok=True)
                overlay_had_original = overlay_target.exists()
                if overlay_had_original:
                    overlay_backup = overlay_target.read_bytes()
                shutil.copy2(staging_path, overlay_target)

            # Run the boot script in a subprocess
            env = os.environ.copy()
            env["PYTHONPATH"] = str(sandbox_path)
            env["AURA_GHOST_BOOT"] = "1" # Flag to skip heavy systems
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(boot_script),
                cwd=str(sandbox_path),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_data, stderr_data = await asyncio.wait_for(process.communicate(), timeout=timeout)
                output = stdout_data.decode()
                error_output = stderr_data.decode()
            except asyncio.TimeoutError:
                process.kill()
                stdout_data, stderr_data = await process.communicate()
                output = stdout_data.decode()
                error_output = stderr_data.decode()
                error_output += "\n[Timeout reached without heartbeat]"

            success = "GHOST_HEARTBEAT_STABLE" in output

            if success:
                logger.info("✅ Ghost Boot SUCCESS: System reached stable state.")
                return True, "Stable boot reached"
            else:
                error_msg = error_output or "Timeout reached without heartbeat"
                logger.error("❌ Ghost Boot FAILED: %s", error_msg)
                return False, f"Boot failure: {error_msg}"

        except Exception as e:
            record_degradation('boot_validator', e)
            logger.error("Ghost Boot execution error: %s", e)
            return False, str(e)
        finally:
            if overlay_target is not None:
                try:
                    if overlay_had_original and overlay_backup is not None:
                        overlay_target.write_bytes(overlay_backup)
                    elif overlay_target.exists():
                        overlay_target.unlink()
                except Exception as e:
                    record_degradation('boot_validator', e)
                    logger.debug("Failed to restore ghost boot overlay: %s", e)
            if boot_script.exists():
                try:
                    boot_script.unlink()
                except Exception as e:
                    record_degradation('boot_validator', e)
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
except Exception as e:
    import traceback
    logger.error("Ghost: BOOT CRASH: %s", e)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)
