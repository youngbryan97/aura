import logging
from typing import Dict, Any, Optional
from core.tasks import execute_skill_task
from core.security.input_sanitizer import input_sanitizer

logger = logging.getLogger("Aura.Router")

class IntentRouter:
    """
    Cognitive Dispatcher for Aura Zenith.
    Decides whether a skill execution should be:
    1. SYNC: High-priority, low-latency (e.g. status check).
    2. ASYNC: Background task (e.g. data forging, reasoning).
    """
    
    # Skills that MUST be backgrounded due to high complexity/latency
    ASYNC_ONLY_SKILLS = {
        "forge_skill", 
        "run_self_audit", 
        "complex_reasoning",
        "generate_full_report"
    }

    async def route_execution(self, skill_name: str, params: dict, engine: Any) -> dict:
        """
        Intelligently routes the execution request.
        """
        # Phase 8 API Hardening: Rigorous input sanitization
        sanitized_params, is_safe = input_sanitizer.validate_params(params)
        if not is_safe:
            logger.error("Security blocks execution of '%s' due to malicious parameters.", skill_name)
            return {"ok": False, "status": "REJECTED_SECURITY", "message": f"Execution of {skill_name} blocked by InputSanitizer."}

        if skill_name in self.ASYNC_ONLY_SKILLS:
            logger.info("⚡ Routing '%s' to background worker (Celery)", skill_name)
            # Dispatch to Celery
            task = execute_skill_task.delay(skill_name, sanitized_params)
            return {
                "ok": True, 
                "status": "QUEUED", 
                "task_id": task.id, 
                "message": f"Skill '{skill_name}' dispatched to background substrate."
            }
        
        # Default: Synchronous execution within the API loop
        logger.info("🧠 Executing '%s' synchronously", skill_name)
        return await engine.execute(skill_name, sanitized_params)
