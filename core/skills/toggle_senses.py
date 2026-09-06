"""Toggle Senses Skill
Enables or Disables sensory perception services (Vision/Hearing).
"""
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from core.config import config
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.resource_observation import get_resource_observer
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.skills.base_skill import BaseSkill
from core.thought_stream import get_emitter

logger = logging.getLogger("Skills.ToggleSenses")

def _sense_state_dir() -> Path:
    path = config.paths.data_dir / "senses"
    path.mkdir(parents=True, exist_ok=True)
    return path

def _get_pid_file(sense_name: str) -> str:
    return str(_sense_state_dir() / f"{sense_name}.pid")

def _save_pid(sense_name: str, pid: int):
    get_file_write_gateway().write_text(
        _get_pid_file(sense_name),
        str(pid),
        source="skills.toggle_senses.pid",
    )

def _load_pid(sense_name: str) -> Optional[int]:
    path = _get_pid_file(sense_name)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return int(f.read().strip())
        except (OSError, IOError):
            return None
    return None

def _clear_pid(sense_name: str):
    path = _get_pid_file(sense_name)
    if os.path.exists(path):
        try: os.remove(path)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('toggle_senses', e)
            from core.sovereign.errors import SensesError
            raise SensesError(f"Failed to clear PID for {sense_name}: {e}", context={"sensor": sense_name})

class SenseController:
    """Manages spawning and kill of sense subprocesses (vision, hearing, etc.)"""
    
    def __init__(self):
        self._processes: Dict[str, subprocess.Popen] = {}
        # Load from disk and clean up zombies
        for sense in ["vision", "hearing", "vocal"]:
            pass  # no-op: intentional

def _is_pid_alive(pid: int) -> bool:
    """Check if a PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False

class ToggleParams(BaseModel):
    sense: Literal["vision", "hearing"] = Field(..., description="The sense to toggle.")
    action: Literal["on", "off"] = Field(..., description="Action to perform.")
    pid: Optional[int] = Field(None, description="Specific PID to stop (optional).")

class ToggleSensesSkill(BaseSkill):
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    name = "toggle_senses"
    description = "Turn 'eyes' (vision) or 'ears' (hearing) on/off."
    input_model = ToggleParams
    
    def __init__(self):
        super().__init__()
        # Load from disk and clean up zombies
        self._script_pids = {}
        for sense in ["vision", "hearing"]:
            pid = _load_pid(sense)
            if pid:
                if not _is_pid_alive(pid):
                    logger.warning("Cleaning up stale %s PID %s", sense, pid)
                    _clear_pid(sense)
                else:
                    self._script_pids[sense] = pid
        
    async def execute(self, params: ToggleParams, context: Dict[str, Any]) -> Dict[str, Any]:
        # Legacy support
        if isinstance(params, dict):
             try:
                 params = ToggleParams(**params)
             except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                 record_degradation('toggle_senses', e)
                 return {"ok": False, "error": f"Invalid input: {e}"}

        sense = params.sense
        action = params.action
        
        if sense == "vision":
            script = "senses/vision_service.py"
        elif sense == "hearing":
            script = "senses/audio_service.py"
        else:
            # Should be caught by Pydantic Literal, but safe guard
            return {"ok": False, "error": f"Unknown sense: {sense}"}
            
        if action == "on":
            try:
                script_path = (config.paths.project_root / script).resolve()
                if not script_path.exists():
                    return {"ok": False, "error": f"Sense service script not found: {script_path}"}

                process = get_subprocess_gateway().spawn(
                    [sys.executable, str(script_path)],
                    cwd=str(config.paths.project_root),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    source=f"skills.toggle_senses.{sense}",
                    start_new_session=True,
                    accelerator_capability="none",
                )
                pid = int(process.pid)
                self._script_pids[sense] = pid
                _save_pid(sense, pid) 
                get_emitter().emit("Senses", f"👁️ {sense.title()} Activated (PID: {pid})", level="success")
                return {"ok": True, "message": f"{sense} activated.", "pid": pid}
            except (subprocess.SubprocessError, OSError) as e:
                record_degradation('toggle_senses', e)
                logger.error("Failed to start %s: %s", sense, e)
                return {"ok": False, "error": f"Failed to start {sense}: {e}"}
                
        elif action == "off":
            target_pid = params.pid or self._script_pids.get(sense)
            if target_pid is not None:
                try:
                    os.kill(int(target_pid), signal.SIGTERM)
                except ProcessLookupError:
                    # Already gone — the desired end state, report it honestly.
                    self._script_pids.pop(sense, None)
                    _clear_pid(sense)
                    return {"ok": True, "message": f"{sense} was already stopped (PID {target_pid} not running)."}
                except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
                    record_degradation('toggle_senses', e)
                    logger.error("Failed to stop %s (PID %s): %s", sense, target_pid, e)
                    return {"ok": False, "error": f"Failed to stop {sense}: {e}"}

                # SIGTERM is a request, not a guarantee — verify the process
                # actually exited before claiming the sense is off.
                stopped = await self._wait_for_exit(int(target_pid), timeout_s=3.0)
                self._script_pids.pop(sense, None)
                _clear_pid(sense)
                if stopped:
                    get_emitter().emit("Senses", f"👁️ {sense.title()} Deactivated.", level="warning")
                    return {"ok": True, "message": f"{sense} deactivated (PID {target_pid} stopped)."}
                return {
                    "ok": False,
                    "error": (
                        f"Sent SIGTERM to {sense} (PID {target_pid}) but it is still "
                        "running after 3s — it may be ignoring the signal."
                    ),
                }
            else:
                logger.warning("No tracked PID for %s; cannot stop.", sense)
                return {"ok": False, "error": f"No tracked PID for {sense}. Provide 'pid' parameter."}
            
        return {"ok": False, "error": "Invalid action."}

    @staticmethod
    async def _wait_for_exit(pid: int, *, timeout_s: float) -> bool:
        """Poll until the pid is truly gone (or a reap-pending zombie).

        os.kill(pid, 0) succeeds on zombies, so signal-probing alone would
        misreport a terminated-but-unreaped child as still running.
        """
        import asyncio

        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            process = await asyncio.to_thread(get_resource_observer().process, pid)
            if process is None:
                return True
            if process.status.lower() in {"dead", "zombie"}:
                return True
            await asyncio.sleep(0.1)
        return False
