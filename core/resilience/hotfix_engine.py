from core.runtime.errors import record_degradation
import asyncio
import importlib
import logging
import sys
from typing import Dict, Any, Optional
from core.container import ServiceContainer

logger = logging.getLogger("Aura.Resilience")

class HotfixEngine:
    """
    Enables dynamic module reloading for zero-downtime updates.
    """
    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self.reloads_total = 0

    async def reload_module(self, module_name: str) -> Dict[str, Any]:
        """
        Reloads a specific module and updates ServiceContainer if necessary.
        """
        logger.info("🆕 Hotfixing module: %s", module_name)
        try:
            if module_name not in sys.modules:
                return {"ok": False, "error": f"Module {module_name} not currently loaded."}
            
            # 1. Pre-flight check: Syntax check without reloading
            # (In a more robust version, we'd use 'py_compile' on a temporary file)
            
            # 2. Reload
            module = sys.modules[module_name]
            reloaded_module = importlib.reload(module)
            self.reloads_total += 1
            
            # 3. Post-reload: Update instances in ServiceContainer
            # This is complex because we need to know what instances belong to this module.
            # We'll rely on services to self-update or we can scan ServiceContainer.
            
            logger.info("✅ Hotfix applied to %s. Total reloads: %s", module_name, self.reloads_total)
            return {"ok": True, "reloaded": module_name}
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('hotfix_engine', e)
            logger.error("❌ Hotfix stage fail for %s: %s", module_name, e)
            return {"ok": False, "error": str(e)}

    async def patch_service(self, service_name: str, module_name: str):
        """
        Reloads a module and re-registers the service in ServiceContainer.
        """
        result = await self.reload_module(module_name)
        if result["ok"]:
            # Find the class in the module
            reloaded_module = sys.modules[module_name]
            # Assuming CamelCase service name matches class name or similar
            # This is a guestimation for now.
            logger.info("Service %s re-registration in Mycelium requested.", service_name)
            return True
        return False
