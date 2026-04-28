"""
Grounded Search Skill — Ported from gemini-cli/web-search.ts

Uses google-genai SDK to perform real Google Search queries with
grounding metadata, providing inline citations and reducing hallucinations.
"""

from core.runtime.errors import record_degradation
import logging
import os
from typing import Any, Dict

from infrastructure import BaseSkill

logger = logging.getLogger("Skills.GroundedSearch")

class GroundedSearchSkill(BaseSkill):
    name = "grounded_search"
    description = "Searches the web using Google Search API with inline citation grounding."

    async def execute(self, goal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        query = goal.get("params", {}).get("query", goal.get("objective", ""))
        
        if not query:
            return {"ok": False, "error": "No query provided"}

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {
                "ok": False, 
                "error": "GEMINI_API_KEY is not set. Cannot use Google Grounding.",
                "note": "Fallback to standard web_search if needed."
            }

        try:
            # We delay import until runtime to prevent strict dependencies
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=api_key)
            logger.info("Executing grounded search for: %s", query)
            
            response = client.models.generate_content(
                model='gemini-2.5-pro',
                contents=query,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    temperature=0.0
                )
            )

            # Format the text with inline citations if metadata is present
            answer = response.text
            sources = []
            
            # The python SDK typically exposes grounding_metadata if tools were used
            metadata = getattr(response.candidates[0], "grounding_metadata", None)
            if metadata and metadata.grounding_chunks:
                for chunk in metadata.grounding_chunks:
                    if hasattr(chunk, "web"):
                        sources.append({
                            "title": chunk.web.title,
                            "url": chunk.web.uri
                        })
            
            if sources:
                answer += "\n\n### Grounding Sources:\n"
                for i, src in enumerate(sources, 1):
                    answer += f"[{i}] [{src['title']}]({src['url']})\n"

            return {
                "ok": True,
                "answer": answer,
                "sources": sources,
                "note": "Grounded by Google Search"
            }
            
        except ImportError:
            return {"ok": False, "error": "google-genai package not installed (pip install google-genai)"}
        except Exception as e:
            record_degradation('grounded_search', e)
            logger.error("Grounded Search failed: %s", e)
            return {"ok": False, "error": str(e)}
