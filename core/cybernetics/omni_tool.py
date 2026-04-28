from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import logging
import time
import asyncio
from typing import Any, Dict, List, Optional, Callable

class OmniTool:
    """
    [ZENITH] The Omni-Tool Command Interface (Mass Effect inspired).
    A unified router for safe, permissioned, and cooldown-protected tool execution.
    """
    def __init__(self, kernel: Any = None):
        self.kernel = kernel
        self._event_bus = None
        self._cooldowns: Dict[str, float] = {}
        # Consolidation: Unified Execution Logs
        self._execution_logs: Dict[str, List[Dict[str, Any]]] = {}
        self._permissions: Dict[str, bool] = {
            "reboot": False,
            "kernel_patch": False,
            "external_request": True
        }
        # Daemon Supervisor: Proactive Task Management
        self._daemons: Dict[str, Dict[str, Any]] = {}

    async def load(self):
        try:
            from core.event_bus import get_event_bus
            self._event_bus = get_event_bus()
        except ImportError:
            self._event_bus = None
        logger.info("🔋 [OMNI] Omni-Tool Interface ENGAGED. Field actions READY.")

    async def execute_action(self, action_name: str, handler: Callable, *args, **kwargs) -> Any:
        """
        Executes a field action with standardized safety guardrails.
        """
        now = time.time()
        
        # 1. Cooldown Enforcement
        last_run = self._cooldowns.get(action_name, 0.0)
        cooldown_period = kwargs.pop('cooldown', 5.0)
        if now - last_run < cooldown_period:
            logger.warning("⏳ [OMNI] Action '%s' is cooling down.", action_name)
            return {"error": "cooldown_active", "remaining": cooldown_period - (now - last_run)}

        # 2. Permission Validation
        if not self._permissions.get(action_name, True):
            logger.error("🚫 [OMNI] Action '%s' DENIED by system security policy.", action_name)
            return {"error": "permission_denied"}

        # 3. Execution with Error Catching
        try:
            logger.info("🚀 [OMNI] Executing Field Action: %s", action_name)
            self._cooldowns[action_name] = now
            
            result = None
            if asyncio.iscoroutinefunction(handler):
                result = await handler(*args, **kwargs)
            else:
                result = handler(*args, **kwargs)
            
            # Log Success
            logs = self._execution_logs.setdefault(action_name, [])
            logs.append({"ts": now, "status": "success"})
            if len(logs) > 50: logs.pop(0) # Keep last 50
            return result
            
        except Exception as e:
            record_degradation('omni_tool', e)
            logger.error("💥 [OMNI] Action '%s' FAILED: %s", action_name, e)
            # Log Failure
            logs = self._execution_logs.setdefault(action_name, [])
            logs.append({"ts": now, "status": "error", "error": str(e)})
            if len(logs) > 50: logs.pop(0)
            return {"error": str(e)}

    async def spawn_daemon(self, name: str, command: str) -> Dict[str, Any]:
        """[DAEMON] Spawns a proactive background task."""
        if name in self._daemons:
            return {"status": "error", "message": f"Daemon '{name}' already exists."}
        
        logger.info("🕯️ [DAEMON] LIGHTING: %s -> %s", name, command)
        # In a real impl, this would use subprocess.Popen
        # Here we simulate with an async task
        metadata = {
            "name": name,
            "command": command,
            "start_time": time.time(),
            "status": "running"
        }
        self._daemons[name] = metadata
        
        # Simulate outcome after 10 seconds
        async def _run_sim():
            await asyncio.sleep(10)
            self._daemons[name]["status"] = "completed"
            self._daemons[name]["end_time"] = time.time()
            logger.info("✨ [DAEMON] EXTINGUISHED: %s", name)

        get_task_tracker().create_task(_run_sim())
        
        if self._event_bus:
            get_task_tracker().create_task(self._event_bus.publish("core/cybernetics/daemon_spawned", metadata))
            
        return {"status": "spawned", "daemon": metadata}

    def check_daemons(self) -> List[Dict[str, Any]]:
        """[DAEMON] Review all active background outcomes."""
        return list(self._daemons.values())

    def get_status(self) -> Dict[str, Any]:
        return {
            "ready": True,
            "active_cooldowns": list(self._cooldowns.keys()),
            "restricted_actions": [k for k, v in self._permissions.items() if not v]
        }

logger = logging.getLogger("Aura.Cybernetics.OmniTool")
