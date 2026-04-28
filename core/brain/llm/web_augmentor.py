from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import time
from typing import Any, Dict, Optional

from ..cognitive_interface import AbstractCognitiveAugmentor


logger = logging.getLogger("Aura.WebAugmentor")

class SovereignWebAugmentor(AbstractCognitiveAugmentor):
    """Gives Aura real-time internet awareness without blocking her thinking cycles.
    Maintains a 'World Context' cache that updates in the background.
    """

    def __init__(self, search_skill=None):
        self.search_skill = search_skill
        self.world_context = "Scanning the horizon... No fresh data yet."
        self.last_update = 0
        self.update_interval = 3600 # 1 hour default
        self._lock = asyncio.Lock()
        self._is_updating = False

    def prepare_context(self, objective: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if we need to trigger an update based on objective."""
        # Reactive Update: If the user asks about 'now', 'today', or 'news'
        keywords = ["now", "today", "news", "current", "latest"]
        if any(w in objective.lower() for w in keywords):
            if time.time() - self.last_update > 300: # 5 min cooldown for reactive
                get_task_tracker().create_task(self.refresh_world_state())
        
        return context

    def enrich_prompt(self, system_prompt: str, context: Dict[str, Any]) -> str:
        """Inject the World State into the system prompt."""
        world_block = f"""
[WORLD STATE]
Current Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
Internet Awareness:
{self.world_context}
[END WORLD STATE]
"""
        return system_prompt + "\n" + world_block

    async def refresh_world_state(self, force: bool = False):
        """Perform a background search for top news/current events."""
        if self._is_updating and not force:
            return
            
        async with self._lock:
            self._is_updating = True
            logger.info("🌐 SovereignWebAugmentor: Refreshing world state...")
            try:
                from core.container import ServiceContainer
                registry = ServiceContainer.get("capability_engine", default=None)
                
                if not self.search_skill:
                    # Fix: Use 'search_web' to match web_search_skill.py
                    self.search_skill = registry.get("search_web") if registry else None
                
                if registry:
                    # Query for top headlines via the unified CapabilityEngine path
                    result = await registry.execute(
                        "search_web",
                        {"query": "top world news today", "deep": True, "retain": True},
                        {"origin": "world_monitor"},
                    )
                    if result.get("ok"):
                        content = (
                            result.get("answer")
                            or result.get("summary")
                            or result.get("message", "")
                        )
                        citations = list(result.get("citations") or [])
                        source_lines = "\n".join(
                            f"- {item.get('title', '')}: {item.get('url', '')}"
                            for item in citations[:3]
                        )
                        self.world_context = f"{content[:1200]}\n{source_lines}".strip()
                        self.last_update = time.time()
                        logger.info("✅ World state updated.")
                    else:
                        logger.warning("Failed to refresh world state: %s", result.get("error"))
                else:
                    logger.warning("WebSearchSkill not available for augmentor.")
            except Exception as e:
                record_degradation('web_augmentor', e)
                logger.error("Error refreshing world state: %s", e)
            finally:
                self._is_updating = False

    def post_think_hook(self, thought: Any, context: Dict[str, Any]):
        """Analyze if Aura's thought suggests a need for deeper research."""
        pass  # no-op: intentional
