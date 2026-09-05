import logging
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service

logger = logging.getLogger("Brain.Composer")

class ComposerNode:
    """
    Local multimodal composition planner.

    The current implementation derives an image-generation description from
    the resident visual context. It does not apply an image transformation.
    """
    
    def __init__(self, container: Any | None = None):
        self._container = container
        self.vision_buffer = None
        self.capability_engine = None
        self._is_setup = False

    def _setup(self) -> None:
        if self._is_setup:
            return
        try:
            if self._container is not None and hasattr(self._container, "get"):
                self.vision_buffer = self._container.get("continuous_vision", default=None)
                self.capability_engine = self._container.get("capability_engine", default=None)
            else:
                self.vision_buffer = get_runtime_service("continuous_vision", default=None)
                self.capability_engine = get_runtime_service("capability_engine", default=None)
            self._is_setup = True
            logger.info("🎨 Composer Node Online (Style Planning Enabled).")
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation('composer_node', e)
            logger.error("Composer setup failed: %s", e)

    async def stylize_desktop(self, style_prompt: str) -> dict[str, Any]:
        """
        Derive a style-transfer plan from the current desktop observation.

        The returned receipt distinguishes planning from an applied media
        effect so callers cannot report a completed transformation.
        """
        self._setup()
        
        if not self.vision_buffer:
            return {"ok": False, "error": "Continuous Vision not available."}
            
        # 1. Capture current frame from buffer
        frames = self.vision_buffer.frame_buffer
        if not frames:
            return {"ok": False, "error": "No frames captured."}
            
        logger.info("🎭 Planning desktop style treatment: '%s'", style_prompt)
        
        try:
            # Ground the plan in Aura's resident visual-context service.
            description = await self.vision_buffer.query_visual_context(
                f"Describe this screen capture in detail for a style transfer to: {style_prompt}",
                get_runtime_service("cognitive_engine", default=None)
            )
            
            # Evolution 8: Pulse Mycelium
            mycelium = get_runtime_service("mycelium", default=None)
            if mycelium:
                mycelium.pulse_hypha("vision", "composer", success=True)

            return {
                "ok": True,
                "workflow": "visual_context_style_plan",
                "base_description": description,
                "effect_applied": False,
                "requires_image_transform": True,
                "message": f"Prepared a grounded style-transfer plan for: {style_prompt}.",
            }
            
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('composer_node', e)
            logger.error("Stylization workflow failed: %s", e)
            return {"ok": False, "error": str(e)}
