from typing import Any, Dict

from pydantic import BaseModel, Field

from core.skills.web_search import EnhancedWebSearchSkill


class SearchInput(BaseModel):
    query: str = Field(..., description="The search query to execute. Be specific.")

class WebSearchSkill(EnhancedWebSearchSkill):
    """Compatibility alias for the modern resilient web-search skill."""

    name = "search_web"
    description = "Search the open web for information. Use this to find facts, news, or deep dive on topics."
    input_model = SearchInput

    async def execute(self, params: SearchInput, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a web search through the shared modern implementation."""
        if isinstance(params, dict):
            try:
                params = SearchInput(**params)
            except Exception as e:
                return {"ok": False, "error": f"Invalid input: {e}"}

        query = params.query if isinstance(params, SearchInput) else str(params)
        return await super().execute({"query": query}, context)
