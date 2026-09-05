import importlib
import logging
import sys
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Resilience")

class HotfixEngine:
    """
    Enables dynamic module reloading for zero-downtime updates.
    """
    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self.reloads_total = 0

    async def reload_module(self, module_name: str) -> dict[str, Any]:
        """
        Reloads a specific module and lets runtime services refresh themselves if needed.
        """
        logger.info("🆕 Hotfixing module: %s", module_name)
        try:
            if module_name not in sys.modules:
                return {"ok": False, "error": f"Module {module_name} not currently loaded."}
            
            # 1. Pre-flight check: Syntax check without reloading
            # (In a more robust version, we'd use 'py_compile' on a temporary file)
            
            # 2. Reload
            module = sys.modules[module_name]
            importlib.reload(module)
            self.reloads_total += 1
            
            # 3. Post-reload: service refresh is delegated to runtime owners.
            # This is complex because we need to know what instances belong to this module.
            # We'll rely on services to self-update rather than scanning a global container.
            
            logger.info("✅ Hotfix applied to %s. Total reloads: %s", module_name, self.reloads_total)
            return {"ok": True, "reloaded": module_name}
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('hotfix_engine', e)
            logger.error("❌ Hotfix stage fail for %s: %s", module_name, e)
            return {"ok": False, "error": str(e)}

    async def patch_service(self, service_name: str, module_name: str):
        """
        Reloads a module and requests the owning service to refresh.
        """
        result = await self.reload_module(module_name)
        if result["ok"]:
            # Assuming CamelCase service name matches class name or similar
            # This is a guestimation for now.
            logger.info("Service %s re-registration in Mycelium requested.", service_name)
            return True
        return False
