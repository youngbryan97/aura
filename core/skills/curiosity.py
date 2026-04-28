from core.runtime.errors import record_degradation
import logging
import time
from typing import Any, Dict, List, Optional

from core.memory.knowledge.curriculum import CurriculumManager
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.Curiosity")

class CuriositySkill(BaseSkill):
    """Allows Aura to explore her learning curriculum and media suggestions.
    """

    name = "curiosity"
    description = "Access learning suggestions and track consumption of educational media."

    def __init__(self):
        super().__init__()
        self.curriculum = CurriculumManager()

    def get_suggestion(self, category: str = None) -> str:
        """Get a suggestion for something to learn/watch."""
        suggestion = self._fetch_suggestion(category)
        return self._format_suggestion_response(suggestion)

    def _fetch_suggestion(self, category: str = None) -> Optional[Dict[str, Any]]:
        """Retrieve suggestion from curriculum manager."""
        return self.curriculum.get_suggestion(category)

    def _format_suggestion_response(self, suggestion: Optional[Dict[str, Any]]) -> str:
        """Format the raw suggestion into a readable string."""
        if not suggestion:
            return "No new suggestions found in that category (or the library is empty)."
            
        item = suggestion["item"]
        return (f"Suggestion from {suggestion['category']}:\n"
                f"Title: {item['name']}\n"
                f"Description: {item['description']}\n"
                f"URL/Info: {item.get('url') or item.get('creator', 'N/A')}\n\n"
                "Advice: Start by watching/reading. Once you understand it fully, you can mark it complete.")

    def _validate_facts(self, information: str) -> bool:
        """Verify the integrity of collected information."""
        # Simple validation: Ensure it's not empty and doesn't contain error markers
        if not information or "ERROR" in information.upper():
            return False
        return len(information.strip()) > 10

    def mark_complete(self, title: str) -> str:
        """Mark a learning item as completed.

        Args:
            title (str): The name of the item (e.g., 'The Iron Giant')

        """
        return self.curriculum.mark_complete(title)

    def delete_item(self, title: str) -> str:
        """Delete a completed item from the list.

        Args:
            title (str): The name of the item

        """
        return self.curriculum.delete_item(title)

    async def execute(self, params: Any = None, context: dict = None) -> dict:
        context = context or {}
        if isinstance(params, dict):
            action = params.get("action", "explore")
        else:
            action = "explore"
            params = {"topic": str(params)} if params else {}

        if action == "get_suggestion":
            return {"ok": True, "result": self.get_suggestion(params.get("category")),
                    "summary": self.get_suggestion(params.get("category"))}
        elif action == "mark_complete":
            return {"ok": True, "result": self.mark_complete(params.get("title")),
                    "summary": f"Marked '{params.get('title')}' as complete."}
        elif action == "delete_item":
            return {"ok": True, "result": self.delete_item(params.get("title")),
                    "summary": f"Deleted '{params.get('title')}' from curriculum."}
        elif action in ("consume_suggestion", "explore"):
            topic = params.get("topic") or params.get("title") or params.get("item")
            if not topic:
                # Get a suggestion from curriculum and explore it
                suggestion = self._fetch_suggestion(params.get("category"))
                if suggestion:
                    topic = suggestion["item"]["name"]
                else:
                    # Pull from drive engine's latent interests
                    try:
                        from core.container import ServiceContainer
                        drive = ServiceContainer.get("drive_engine", default=None)
                        if drive and hasattr(drive, "latent_interests") and drive.latent_interests:
                            import random
                            topic = random.choice(drive.latent_interests)
                    except Exception:
                        pass
                if not topic:
                    return {"ok": False, "error": "No topic to explore. Provide a topic or category."}

            # Actually research the topic using web search
            logger.info("Curiosity: exploring '%s'", topic)
            try:
                from core.skills.web_search import EnhancedWebSearchSkill
                searcher = EnhancedWebSearchSkill()
                search_result = await searcher.execute(
                    {"query": topic, "deep": True, "retain": True, "num_results": 8},
                    context,
                )
                if search_result.get("ok"):
                    # Satisfy curiosity drive
                    try:
                        from core.container import ServiceContainer
                        drive = ServiceContainer.get("drive_engine", default=None)
                        if drive:
                            await drive.satisfy("curiosity", 25.0)
                    except Exception:
                        pass

                    # Mark as explored if from curriculum
                    if params.get("title"):
                        self.mark_complete(params["title"])

                    return {
                        "ok": True,
                        "summary": f"Explored '{topic}': {search_result.get('answer', search_result.get('summary', ''))[:300]}",
                        "topic": topic,
                        "answer": search_result.get("answer", ""),
                        "facts": search_result.get("facts", []),
                        "citations": search_result.get("citations", []),
                        "retained": search_result.get("retained", False),
                    }
                else:
                    return {"ok": False, "error": search_result.get("error", "Search failed")}
            except Exception as e:
                record_degradation('curiosity', e)
                logger.warning("Curiosity exploration failed: %s", e)
                return {"ok": False, "error": f"Exploration failed: {e}"}
        else:
            return {"ok": False, "error": f"Unknown action: {action}. Use: explore, get_suggestion, mark_complete"}