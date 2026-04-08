"""Toggle Senses Skill
Enables or Disables sensory perception services (Vision/Hearing).
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill
from core.thought_stream import get_emitter
from core.container import ServiceContainer
import subprocess

from core.config import config

logger = logging.getLogger("Skills.ToggleSenses")

def _sense_state_dir() -> Path:
    path = config.paths.data_dir / "senses"
    path.mkdir(parents=True, exist_ok=True)
    return path

def _get_pid_file(sense_name: str) -> str:
    return str(_sense_state_dir() / f"{sense_name}.pid")

def _save_pid(sense_name: str, pid: int):
    with open(_get_pid_file(sense_name), "w") as f:
        f.write(str(pid))

def _load_pid(sense_name: str) -> Optional[int]:
    path = _get_pid_file(sense_name)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return int(f.read().strip())
        except Exception:
            return None
    return None

def _clear_pid(sense_name: str):
    path = _get_pid_file(sense_name)
    if os.path.exists(path):
        try: os.remove(path)
        except Exception as e:
            from core.errors import SensesError
            raise SensesError(f"Failed to clear PID for {sense_name}: {e}", context={"sensor": sense_name})

class SenseController:
    """Manages spawning and kill of sense subprocesses (vision, hearing, etc.)"""
    
    def __init__(self):
        self._processes: Dict[str, subprocess.Popen] = {}
        # Load from disk and clean up zombies
        for sense in ["vision", "hearing", "vocal"]:
            pass

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
             except Exception as e:
                 return {"ok": False, "error": f"Invalid input: {e}"}

        sense = params.sense
        action = params.action
        
        # Issue 84: Resolve sandbox correctly
        sandbox = ServiceContainer.get("local_sandbox", default=None)
        if not sandbox:
            from core.sovereign.local_sandbox import LocalSandbox
            sandbox = LocalSandbox(sandbox_id="senses_controller")
        
        if sense == "vision":
            script = "senses/vision_service.py"
        elif sense == "hearing":
            script = "senses/audio_service.py"
        else:
            # Should be caught by Pydantic Literal, but safe guard
            return {"ok": False, "error": f"Unknown sense: {sense}"}
            
        if action == "on":
            try:
                pid = sandbox.start_process(script)
                self._script_pids[sense] = pid
                _save_pid(sense, pid) 
                get_emitter().emit("Senses", f"👁️ {sense.title()} Activated (PID: {pid})", level="success")
                return {"ok": True, "message": f"{sense} activated.", "pid": pid}
            except Exception as e:
                logger.error("Failed to start %s: %s", sense, e)
                return {"ok": False, "error": f"Failed to start {sense}: {e}"}
                
        elif action == "off":
            target_pid = params.pid or self._script_pids.get(sense)
            if target_pid is not None:
                try:
                    sandbox.stop_process(int(target_pid))
                    self._script_pids.pop(sense, None)
                    _clear_pid(sense)
                    get_emitter().emit("Senses", f"👁️ {sense.title()} Deactivated.", level="warning")
                    return {"ok": True, "message": f"{sense} deactivated (PID {target_pid} stopped)."}
                except Exception as e:
                    logger.error("Failed to stop %s (PID %s): %s", sense, target_pid, e)
                    return {"ok": False, "error": f"Failed to stop {sense}: {e}"}
            else:
                logger.warning("No tracked PID for %s; cannot stop.", sense)
                return {"ok": False, "error": f"No tracked PID for {sense}. Provide 'pid' parameter."}
            
        return {"ok": False, "error": "Invalid action."}
