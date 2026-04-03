from typing import Dict, Any
from core.skills.base_skill import BaseSkill
import logging

logger = logging.getLogger("Skills.StealthOps")

class StealthOpsSkill(BaseSkill):
    """Skill for autonomous stealth and privacy management.
    Wraps core/privacy_stealth.py for Brain accessibility.
    """
    
    name = "stealth_ops"
    description = "Manage VPN, IP rotation, and metadata scrubbing for anonymity."
    
    def __init__(self):
        super().__init__()
        self.logger = logger
        self.stealth_manager = None
        self._initialize_manager()
        
    def _initialize_manager(self):
        try:
            from core.privacy_stealth import StealthMode
            self.stealth_manager = StealthMode()
            self.logger.info("✓ Stealth manager integrated into skill")
        except Exception as e:
            self.logger.error("Failed to integrate stealth core: %s", e)

    async def execute(self, goal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.stealth_manager:
            return {"ok": False, "error": "core_not_found", "message": "Stealth core module missing."}
            
        action = goal.get("params", {}).get("command", "status")
        
        try:
            if action == "enable":
                server = goal.get("params", {}).get("server")
                success = await self.stealth_manager.enable_stealth(vpn_server=server)
                return {
                    "ok": success, 
                    "status": await self.stealth_manager.get_stealth_status()
                }
                
            elif action == "disable":
                success = await self.stealth_manager.disable_stealth()
                return {
                    "ok": success, 
                    "status": await self.stealth_manager.get_stealth_status()
                }
                
            elif action == "rotate":
                proxy = self.stealth_manager.ip_spoof.rotate_proxy()
                return {
                    "ok": proxy is not None,
                    "proxy": proxy,
                    "status": await self.stealth_manager.get_stealth_status()
                }
                
            elif action == "status":
                return {
                    "ok": True, 
                    "status": await self.stealth_manager.get_stealth_status()
                }
                
            else:
                return {"ok": False, "error": "invalid_command", "message": f"Unknown command: {action}"}
                
        except Exception as e:
            self.logger.error("Stealth execution error: %s", e)
            return {"ok": False, "error": str(e)}