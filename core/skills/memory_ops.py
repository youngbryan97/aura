from core.runtime.atomic_writer import atomic_write_text
import logging
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field
from core.config import config
from core.container import ServiceContainer
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.MemoryOps")


class MemoryOpsInput(BaseModel):
    action: str = Field(
        ...,
        description=(
            "Memory action. Supports canonical Letta verbs "
            "('core_append', 'core_replace', 'archival_insert', 'archival_search') "
            "plus runtime aliases like 'remember' and 'recall'."
        ),
    )
    block: Optional[str] = Field(None, description="The Core Memory block name (e.g., 'persona', 'user') for core_* ops.")
    content: Optional[str] = Field(None, description="Data to append, insert, or replace.")
    old_content: Optional[str] = Field(None, description="Exact prior string to replace. Used only in 'core_replace'.")
    query: Optional[str] = Field(None, description="Search term for 'archival_search'.")


class MemoryOpsSkill(BaseSkill):
    name = "memory_ops"
    description = "Hierarchical memory management (RAM vs Disk) modeled after Letta. Edit Core memory blocks or search Archival storage."
    input_model = MemoryOpsInput
    _ACTION_ALIASES = {
        "remember": "archival_insert",
        "memorize": "archival_insert",
        "store": "archival_insert",
        "save": "archival_insert",
        "recall": "archival_search",
        "search": "archival_search",
        "query": "archival_search",
        "read": "archival_search",
    }
    
    def __init__(self):
        super().__init__()
        # Initialize MemFS (Memory File System) in the workspace
        self.mem_fs_dir = Path(getattr(config.paths, "base_dir", ".")) / ".aura" / "memfs"
        self.mem_fs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize default blocks if missing
        self.core_blocks = ["persona", "user", "system"]
        for block in self.core_blocks:
            path = self.mem_fs_dir / f"{block}.txt"
            if not path.exists():
                atomic_write_text(path, f"// Core Memory Block: {block}\n", encoding="utf-8")

    async def execute(self, params: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(params, dict):
            try:
                params = MemoryOpsInput(**params)
            except Exception as e:
                return {"ok": False, "error": f"Invalid input: {e}"}

        action = self._normalize_action(params.action)
        if action == "archival_insert" and not params.content and params.query:
            params = params.model_copy(update={"content": params.query})
        elif action == "archival_search" and not params.query and params.content:
            params = params.model_copy(update={"query": params.content})
        
        try:
            if action.startswith("core_"):
                return await self._execute_core_memory(params, context, action)
            elif action.startswith("archival_"):
                return await self._execute_archival_memory(params, context, action)
            else:
                return {"ok": False, "error": f"Unknown memory action: {action}"}
        except Exception as e:
            logger.error("MemoryOps failed: %s", e)
            return {"ok": False, "error": str(e)}

    @classmethod
    def _normalize_action(cls, action: Any) -> str:
        lowered = str(action or "").strip().lower()
        return cls._ACTION_ALIASES.get(lowered, lowered)

    @staticmethod
    def _resolve_memory_facade(context: Dict[str, Any]) -> Any:
        return context.get("memory_facade") or ServiceContainer.get("memory_facade", default=None)

    async def _execute_core_memory(self, params: MemoryOpsInput, context: Dict[str, Any], action: str) -> Dict[str, Any]:
        """RAM: Immediate context window blocks."""
        block = params.block or "user"
        if not block.isalnum() and "_" not in block:
            return {"ok": False, "error": "Invalid block name. Must be alphanumeric."}
            
        block_path = self.mem_fs_dir / f"{block}.txt"
        
        if action == "core_append":
            if not params.content:
                return {"ok": False, "error": "Missing 'content' to append."}
            with open(block_path, "a", encoding="utf-8") as f:
                f.write(params.content + "\n")
            return {"ok": True, "summary": f"Appended to core memory block '{block}'."}

        elif action == "core_replace":
            if not params.content or not params.old_content:
                return {"ok": False, "error": "Missing 'content' or 'old_content' for replacing."}
            
            with open(block_path, "r", encoding="utf-8") as f:
                data = f.read()
                
            if params.old_content not in data:
                return {"ok": False, "error": f"Text to replace not found in block '{block}'."}
                
            new_data = data.replace(params.old_content, params.content)
            with open(block_path, "w", encoding="utf-8") as f:
                f.write(new_data)
                
            return {"ok": True, "summary": f"Replaced content in core memory block '{block}'."}
            
        return {"ok": False, "error": f"Unknown core action: {action}"}

    async def _execute_archival_memory(self, params: MemoryOpsInput, context: Dict[str, Any], action: str) -> Dict[str, Any]:
        """Disk: Long-term archival Vector / DB storage."""
        memory_facade = self._resolve_memory_facade(context)
        if not memory_facade:
            return {"ok": False, "error": "Archival backend (memory_facade) is not wired to context."}

        if action == "archival_insert":
            if not params.content:
                return {"ok": False, "error": "Missing 'content' to archive."}
            
            try:
                # Assuming standard facade pattern
                import asyncio
                if hasattr(memory_facade, "add_memory"):
                    # Use async or sync dynamically
                    res = memory_facade.add_memory(params.content, metadata={"source": "archival_insert"})
                    if hasattr(res, "__await__"):
                        await res
                elif hasattr(memory_facade, "update_semantic_async"):
                    await memory_facade.update_semantic_async("archival_" + str(len(params.content)), params.content)
                else:
                    return {"ok": False, "error": "Facade missing insertion capability."}
                return {"ok": True, "summary": "Committed to archival storage."}
            except Exception as e:
                return {"ok": False, "error": f"Archival insertion failed: {e}"}

        elif action == "archival_search":
            if not params.query:
                return {"ok": False, "error": "Missing 'query' to search."}
            
            try:
                if hasattr(memory_facade, "search_memories"):
                    res = memory_facade.search_memories(params.query, limit=5)
                    results = await res if hasattr(res, "__await__") else res
                elif hasattr(memory_facade, "search"):
                    res = memory_facade.search(params.query, limit=5)
                    results = await res if hasattr(res, "__await__") else res
                elif hasattr(memory_facade, "query_memory"):
                    res = memory_facade.query_memory(params.query, limit=5)
                    results = await res if hasattr(res, "__await__") else res
                else:
                    return {"ok": False, "error": "Facade missing search capability."}
                
                # Format Letta style
                formatted = []
                for item in results or []:
                    if isinstance(item, dict):
                        score = float(item.get("score", 0) or 0)
                        content = item.get("content") or item.get("text")
                        if content:
                            formatted.append(f"[{score:.2f}] {content}")
                return {
                    "ok": True, 
                    "results": formatted if formatted else ["No archival memories found."],
                    "summary": f"Found {len(formatted)} artifacts."
                }
            except Exception as e:
                return {"ok": False, "error": f"Archival search failed: {e}"}

        return {"ok": False, "error": f"Unknown archival action: {action}"}
