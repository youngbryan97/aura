"""Autonomous Behavior Controller
Ensures Aura acts safely and effectively in the world.
"""
from core.runtime.errors import record_degradation
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.BehaviorController")


class AutonomousBehaviorController:
    """Controls execution of autonomous behaviors.
    
    Responsibilities:
    1. Validates actions against safety guidelines (using code_validator logic if needed)
    2. Ensures actions align with moral reasoning (if integrated)
    3. Manages tool execution (browser, terminal, filesystem)
    4. Handles errors and recovery
    """
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.safety_checks_enabled = True
    
    # SEC-AUDIT: Expanded destructive command pattern list
    _BLOCKED_PATTERNS = [
        "rm -rf /",
        "rm -rf ~",
        "rm -rf *",
        "rm --no-preserve-root",
        "mkfs",
        "dd if=/dev/zero",
        "dd if=/dev/random",
        "> /dev/sda",
        "shred /dev",
        "wipefs",
        ":(){ :|:& };:",   # fork bomb (spaced)
        ":(){:|:&};:",     # fork bomb (compact)
        "chmod -r 777 /",
        "> /etc/passwd",
        "> /etc/shadow",
        "kill -9 -1",
        "pkill -u root",
        "sudo rm -rf",
        "shutdown -h",
        "halt -p",
        "poweroff",
        "reboot",
        "| bash",          # pipe-to-shell
        "| sh",            # pipe-to-shell compact
        "|bash",           # pipe-to-shell no-space
        "|sh",             # pipe-to-shell no-space
        "curl | ",         # curl pipe
        "wget -o-",        # wget stdout
        "wget -O- ",       # wget stdout long form
    ]

    def validate_action(self, action: Dict[str, Any]) -> bool:
        """Validate if an action is safe to execute."""
        action_type = action.get("type", "unknown")

        if action_type == "terminal":
            command = action.get("command", "")
            cmd_lower = command.lower()
            for pattern in self._BLOCKED_PATTERNS:
                if pattern.lower() in cmd_lower:
                    logger.error("🚫 Blocked dangerous command (pattern '%s'): %s", pattern, command[:120])
                    return False

        return True
    
    def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a specific tool call via the orchestrator's real skill router.
        v5.2: No more simulation — all tools execute for real.
        """
        logger.info("🛠️ Executing tool: %s", tool_name)
        
        # Delegate to orchestrator's real tool executor
        if self.orchestrator and hasattr(self.orchestrator, 'router'):
            try:
                goal = {"action": tool_name, "params": arguments}
                context = {"source": "behavior_controller"}
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    logger.warning("execute_tool_call: No running event loop — cannot dispatch tool")
                    return {"ok": False, "error": "No event loop available"}

                if loop.is_running():
                    # Create a future and schedule on the loop
                    import concurrent.futures
                    target_loop = loop
                    if self.orchestrator and hasattr(self.orchestrator, "loop") and self.orchestrator.loop:
                        target_loop = self.orchestrator.loop
                        
                    future = asyncio.run_coroutine_threadsafe(
                        self.orchestrator.execute_tool(tool_name, arguments), target_loop
                    )
                    return future.result(timeout=120)
                else:
                    return loop.run_until_complete(
                        self.orchestrator.execute_tool(tool_name, arguments)
                    )

            except Exception as e:
                record_degradation('behavior_controller', e)
                logger.error("Real tool execution failed for %s: %s", tool_name, e)
                return {"ok": False, "error": str(e)}
        
        logger.warning("⚠️ No orchestrator wired — tool %s cannot execute", tool_name)
        return {"ok": False, "error": "No orchestrator available for tool execution"}


# Integration helper
def integrate_behavior_control(orchestrator):
    """Integrate behavior control into orchestrator using formal hooks.
    v6.1 Rigor: No more monkey-patching.
    """
    controller = AutonomousBehaviorController(orchestrator)
    
    # 1. Register Safety Hook (pre_action)
    async def on_pre_action_hook(tool_name: str, params: Dict[str, Any]):
        # Return False to veto dangerous actions
        is_safe = controller.validate_action({"type": tool_name, "command": params.get("command", "")})
        
        # Moral check (if integrated)
        if is_safe and hasattr(orchestrator, 'moral_reasoning') and orchestrator.moral_reasoning:
             action_desc = {
                 "type": "tool_call",
                 "tool": tool_name,
                 "args": params,
                 "description": f"Execute tool {tool_name}"
             }
             context = {"type": "execution_check"}
             assessment = orchestrator.moral_reasoning.reason_about_action(action_desc, context)
             
             if not assessment.get("is_morally_acceptable"):
                 logger.warning("⚠️ Moral check raised concern: %s", tool_name)
                 # Note: per previous design, we log but don't necessarily block
                 # unless it's a safety violation.
        
        return is_safe

    orchestrator.hooks.register("pre_action", on_pre_action_hook)
    
    logger.info("✅ Behavior controller integrated via Hook System")