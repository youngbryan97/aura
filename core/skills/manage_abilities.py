import logging
from typing import Any, Dict
from pydantic import BaseModel, Field
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Aura.SkillGovernor")

class ManageAbilitiesInput(BaseModel):
    action: str = Field(..., description="Must be either 'activate' or 'deactivate'")
    skill_name: str = Field(..., description="The exact name of the skill from your Subconscious Index")

class ManageAbilitiesSkill(BaseSkill):
    """Activate or deactivate specific skills to manage your memory and thermal load.
    Deactivate heavy skills when not in use.
    """
    name = "ManageAbilities"
    description = "Manage your cognitive toolset. Activate dormant tools to use them, deactivate when finished."
    input_model = ManageAbilitiesInput
    metabolic_cost = 0
    is_core_personality = True

    async def execute(self, params: ManageAbilitiesInput, context: Dict[str, Any]) -> Dict[str, Any]:
        action = params.action.lower()
        skill_name = params.skill_name
        
        from core.container import ServiceContainer
        engine = ServiceContainer.get("capability_engine", default=None)
        
        if not engine:
            return {"ok": False, "error": "CapabilityEngine unavailable."}

        # Metabolic Override Protection
        metabolism = ServiceContainer.get("metabolic_monitor", default=None)
        if action == "activate" and metabolism:
            health = metabolism.get_current_metabolism().health_score
            skill_meta = engine.get(skill_name)
            if skill_meta:
                cost = skill_meta.metabolic_cost
                # Fail fast if system is too hot for heavy tools
                if health < 0.7 and cost >= 2:
                    logger.warning("🛡️ METABOLIC BLOCK: Refused activation of %s due to thermal stress.", skill_name)
                    return {
                        "ok": False, 
                        "error": f"AUTONOMIC BLOCK: Hardware thermal load is too high (Health: {health*100:.0f}%). Cannot supply compute power to heavy skill '{skill_name}'. Prioritize communication."
                    }

        if action == "activate":
            success = engine.activate_skill(skill_name)
            if success:
                logger.info("🧠 Aura activated skill: %s", skill_name)
                return {"ok": True, "message": f"SYSTEM: {skill_name} is now AWAKE and injected into your active tools. You MUST use it in your very next response if needed. Remember to deactivate it when finished."}
            return {"ok": False, "message": f"Skill '{skill_name}' not found or activation failed."}
            
        elif action == "deactivate":
            success = engine.deactivate_skill(skill_name)
            if success:
                logger.info("💤 Aura put skill to sleep: %s", skill_name)
                return {"ok": True, "message": f"{skill_name} is now DORMANT. Memory freed."}
            return {"ok": False, "message": f"Skill '{skill_name}' is either not active or is a CORE tool."}
            
        return {"ok": False, "message": "Invalid action. Use 'activate' or 'deactivate'."}