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

    async def execute(self, goal: dict, context: dict) -> dict:
        params = goal.get("params", {})
        action = params.get("action")
        
        if action == "get_suggestion":
            return {"ok": True, "result": self.get_suggestion(params.get("category"))}
        elif action == "mark_complete":
            return {"ok": True, "result": self.mark_complete(params.get("title"))}
        elif action == "delete_item":
            return {"ok": True, "result": self.delete_item(params.get("title"))}
        elif action == "consume_suggestion":
            # Alias for mark_complete, as the Brain often hallucinates this action name
            title = params.get("title") or params.get("item")
            if not title: return {"ok": False, "error": "Title required for consume_suggestion"}
            return {"ok": True, "result": self.mark_complete(title)}
        else:
            return {"ok": False, "error": f"Unknown action: {action}"}