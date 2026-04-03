# core/brain/llm/router_interface.py
"""
Concrete implementation of LLMInterface using IntelligentLLMRouter.
"""

import asyncio
from typing import Any, Dict, Optional
from core.brain.llm_interface import LLMInterface
from core.brain.llm.llm_router import IntelligentLLMRouter, LLMTier

class RouterLLMInterface(LLMInterface):
    def __init__(self, router: IntelligentLLMRouter, tier: Optional[LLMTier] = None):
        super().__init__()
        self.router = router
        self.tier = tier

    async def generate(self, prompt: str, **opts) -> str:
        """Async call to the router's think method."""
        tier = opts.get("tier", self.tier)
        priority = opts.get("priority", 1.0)
        
        # Bridge generate options to router think options
        return await self.router.think(
            prompt,
            prefer_tier=tier,
            priority=priority,
            **opts
        )

    def generate_sync(self, prompt: str, **opts) -> str:
        """Sync wrapper using the event loop."""
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                # This should ideally be called in an async context,
                # but if forced sync, we try to run in a thread or new loop
                import threading
                result = []
                def _run():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result.append(new_loop.run_until_complete(self.generate(prompt, **opts)))
                
                t = threading.Thread(target=_run)
                t.start()
                t.join()
                return result[0]
            else:
                return loop.run_until_complete(self.generate(prompt, **opts))
        except Exception:
            # Last resort
            new_loop = asyncio.new_event_loop()
            return new_loop.run_until_complete(self.generate(prompt, **opts))
