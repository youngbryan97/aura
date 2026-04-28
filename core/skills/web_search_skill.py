from core.runtime.errors import record_degradation
from typing import Any, Dict

from pydantic import BaseModel, Field

from core.skills.web_search import EnhancedWebSearchSkill


class SearchInput(BaseModel):
    query: str = Field(..., description="The search query to execute. Be specific.")
    deep: bool = Field(False, description="Whether to fetch and synthesize multiple result pages.")
    num_results: int = Field(5, ge=1, le=20, description="Number of results to return.")
    retain: bool | None = Field(None, description="Whether Aura should retain what she learned.")

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
                record_degradation('web_search_skill', e)
                return {"ok": False, "error": f"Invalid input: {e}"}

        if isinstance(params, SearchInput):
            payload = params.model_dump(exclude_none=True)
        else:
            payload = {"query": str(params)}
        return await super().execute(payload, context)
