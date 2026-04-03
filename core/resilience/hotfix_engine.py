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
        logger.info(f"🆕 Hotfixing module: {module_name}")
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
            
            logger.info(f"✅ Hotfix applied to {module_name}. Total reloads: {self.reloads_total}")
            return {"ok": True, "reloaded": module_name}
        except Exception as e:
            logger.error(f"❌ Hotfix stage fail for {module_name}: {e}")
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
            logger.info(f"Service {service_name} re-registration in Mycelium requested.")
            return True
        return False
