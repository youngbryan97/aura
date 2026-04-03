import logging
from typing import Any, Dict
from core.skills.base_skill import BaseSkill
from core.container import ServiceContainer

logger = logging.getLogger(__name__)

class VisualContextSkill(BaseSkill):
    """
    Skill that allows Aura to query her rolling visual buffer for real-time spatial awareness.
    Provides 'Gemini Live' style screen-awareness.
    """
    
    name = "query_visual_context"
    description = "Analyze the current rolling visual buffer (last 3 frames of screen/camera) to understand visual context."
    
    async def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = params.get("prompt", "Describe what is currently happening on the screen.")
        
        vision = ServiceContainer.get("continuous_vision", default=None)
        if not vision:
            return {"ok": False, "error": "Visual Sensory Buffer is offline."}
            
        brain = ServiceContainer.get("cognitive_engine", default=None)
        if not brain:
            return {"ok": False, "error": "Cognitive Engine is unavailable for visual reasoning."}
            
        try:
            # Query the buffer
            from core.brain.cognitive_engine import ThinkingMode
            analysis = await vision.query_visual_context(prompt=prompt, brain=brain, mode=ThinkingMode.QUICK)
            return {
                "ok": True,
                "analysis": analysis,
                "message": f"I've analyzed my visual field: {analysis}"
            }
        except Exception as e:
            logger.error(f"VisualContextSkill failed: {e}")
            return {"ok": False, "error": str(e)}
