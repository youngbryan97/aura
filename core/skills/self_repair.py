# skills/self_repair.py
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict

from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.SelfRepair")

class SelfRepairSkill(BaseSkill):
    """Skill for autonomous self-healing of the agent's own tools and modules.
    Can analyze errors, locate files, and apply corrections.
    """

    name = "self_repair"
    description = "Audit and repair broken skills or system components."
    inputs = {
        "component": "The name of the broken component or skill.",
        "error": "The error message or failure pattern observed."
    }
    output = "Repair status and actions taken."

    async def execute(self, goal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        params = goal.get("params", {}) if goal.get("params") else {}
        component = goal.get("component") or params.get("component")
        error = goal.get("error") or params.get("error")
        
        if not component:
            # 0. Robustness: If no component specified, we just report status and don't fail.
            logger.info("Self-Repair invoked without target. Reporting system health.")
            return {
                "ok": True, 
                "status": "active", 
                "message": "Self-Repair system is online. Please specify a component to repair (e.g., 'native_chat').",
                "available_actions": ["repair_component", "scan_system"]
            }
        
        logger.info("Initiating Self-Repair for: %s (Error: %s)", component, error)
        
        # 1. Locate the component
        # We check common paths: core/, skills/, etc.
        target_path = None
        for root in ["core", "skills", "infrastructure"]:
            potential_path = os.path.join(root, f"{component}.py")
            if os.path.exists(potential_path):
                target_path = potential_path
                break
        
        if not target_path:
            # Try searching by name
            logger.info("Target path not found directly. Searching for %s...", component)
            # We skip full search here for safety, but in a real shell Aura could use 'find'
            return {"ok": False, "error": f"Could not locate component: {component}"}

        # 2. Analyze and Propose Fix (Requires Brain)
        brain = context.get("brain")
        if not brain:
             return {"ok": False, "error": "Self-Repair requires cognitive access (brain missing in context)."}

        try:
            with open(target_path, "r") as f:
                content = f.read()
            
            prompt = (
                f"FILE: {target_path}\n"
                f"CONTENT:\n{content}\n\n"
                f"ERROR: {error}\n"
                f"TASK: Identify the bug and provide a minimal Python patch or fix instructions. "
                f"Be precise. If it's a dependency, list the 'pip install' command."
            )
            
            from core.brain.cognitive_engine import ThinkingMode
            fix_thought = await brain.think(prompt, mode=ThinkingMode.CRITICAL)
            fix_content = fix_thought.content
            
            # 3. Apply Fix (Safety Check)
            # SECURITY: Do NOT auto-install packages from LLM output.
            if "pip install" in fix_content:
                package = fix_content.split("pip install")[-1].split()[0]
                # Validate package name
                import re as _re
                if not _re.match(r'^[a-zA-Z0-9_-]+$', package):
                    return {"ok": False, "error": f"Suspicious package name: {package}"}
                logger.warning("Self-Repair: Package '%s' needed. NOT auto-installing (security policy).", package)
                return {
                    "ok": False, 
                    "action": "manual_install_required",
                    "message": f"MISSING DEPENDENCY: '{package}' is required but blocked by security policy. PLEASE RUN: pip install {package}"
                }
            
            # For now, we save the "repair_proposal" to a patch file
            patch_path = os.path.join("data", "repairs", f"repair_{component}_{int(time.time())}.patch")
            os.makedirs(os.path.dirname(patch_path), exist_ok=True)
            with open(patch_path, "w") as f:
                f.write(fix_content)
                
            return {
                "ok": True, 
                "message": f"Repair proposal generated for {component}. Saved to {patch_path}",
                "proposal": fix_content[:200] + "..."
            }
            
        except Exception as e:
            logger.error("Repair attempt failed for %s: %s", component, e)
            return {"ok": False, "error": str(e)}