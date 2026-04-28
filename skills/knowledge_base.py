"""
Knowledge Base Skill — Wholesale Addition (Ported from gemini-cli)

Manages persistent Knowledge Items (KIs). KIs are curated, localized context files
written in Markdown, stored in `~/.aura_knowledge/` by default. They serve as 
immutable or slowly changing foundational guides / playbooks for the agent.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from core.config import config
from infrastructure import BaseSkill

logger = logging.getLogger("Skills.KnowledgeBase")

class KnowledgeBaseSkill(BaseSkill):
    name = "knowledge_base"
    description = "Create, read, and search persistent Markdown Knowledge Items (KIs)."
    inputs = {
        "action": "create | read | search",
        "title": "Title of the knowledge item (for create/read)",
        "content": "Markdown content to save (for create)",
        "query": "Search query text (for search)",
        "summary": "Short 1-2 sentence description (for create)"
    }

    def __init__(self):
        # Allow repo-local config via .aura_knowledge, fallback to global
        self.workspace_knowledge = Path(config.paths.base_dir) / ".aura_knowledge"
        self.global_knowledge = Path.home() / ".aura" / ".aura_knowledge"
        
        # Prefer workspace if initialized, otherwise global
        if self.workspace_knowledge.exists():
            self.store_dir = self.workspace_knowledge
        else:
            self.store_dir = self.global_knowledge
            get_task_tracker().create_task(get_storage_gateway().create_dir(self.store_dir, cause='KnowledgeBaseSkill.__init__'))

    def _slugify(self, text: str) -> str:
        return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')

    def _get_metadata_path(self) -> Path:
        return self.store_dir / "metadata.json"

    def _load_metadata(self) -> Dict[str, Any]:
        meta_path = self._get_metadata_path()
        if meta_path.exists():
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_metadata(self, metadata: Dict[str, Any]):
        with open(self._get_metadata_path(), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    async def execute(self, goal: Dict, context: Dict) -> Dict:
        params = goal.get("params", {})
        action = params.get("action", "").lower()

        if action == "create":
            title = params.get("title")
            content = params.get("content")
            summary = params.get("summary", "No summary provided.")
            
            if not title or not content:
                return {"ok": False, "error": "Missing title or content for create action."}

            slug = self._slugify(title)
            filepath = self.store_dir / f"{slug}.md"
            
            # Write item
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {title}\n\n{content}")
                
            # Update metadata index
            metadata = self._load_metadata()
            metadata[slug] = {
                "title": title,
                "summary": summary,
                "path": str(filepath)
            }
            self._save_metadata(metadata)

            return {
                "ok": True, 
                "summary": f"Created KI: {title}", 
                "path": str(filepath)
            }

        elif action == "read":
            title_or_slug = params.get("title", "")
            slug = self._slugify(title_or_slug)
            filepath = self.store_dir / f"{slug}.md"
            
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                return {"ok": True, "content": content}
            return {"ok": False, "error": f"Knowledge item '{title_or_slug}' not found."}

        elif action == "search":
            query = params.get("query", "").lower()
            metadata = self._load_metadata()
            results = []
            
            # Simple keyword search across summaries and titles
            for slug, info in metadata.items():
                if query in info.get("title", "").lower() or query in info.get("summary", "").lower():
                    results.append(info)
            
            return {
                "ok": True, 
                "results": results, 
                "count": len(results),
                "note": "Use 'read' action to pull full content of interesting items."
            }

        return {"ok": False, "error": f"Unknown action: {action}"}
