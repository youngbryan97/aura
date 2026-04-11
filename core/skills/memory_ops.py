import logging
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field
from core.config import config
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.MemoryOps")


class MemoryOpsInput(BaseModel):
    action: str = Field(
        ...,
        description="Letta-based function: 'core_append', 'core_replace', 'archival_insert', 'archival_search'.",
    )
    block: Optional[str] = Field(None, description="The Core Memory block name (e.g., 'persona', 'user') for core_* ops.")
    content: Optional[str] = Field(None, description="Data to append, insert, or replace.")
    old_content: Optional[str] = Field(None, description="Exact prior string to replace. Used only in 'core_replace'.")
    query: Optional[str] = Field(None, description="Search term for 'archival_search'.")


class MemoryOpsSkill(BaseSkill):
    name = "memory_ops"
    description = "Hierarchical memory management (RAM vs Disk) modeled after Letta. Edit Core memory blocks or search Archival storage."
    input_model = MemoryOpsInput
    
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
                path.write_text(f"// Core Memory Block: {block}\n", encoding="utf-8")

    async def execute(self, params: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(params, dict):
            try:
                params = MemoryOpsInput(**params)
            except Exception as e:
                return {"ok": False, "error": f"Invalid input: {e}"}

        action = params.action.lower()
        
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
        memory_facade = context.get("memory_facade")
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
                else:
                    return {"ok": False, "error": "Facade missing 'search_memories' capability."}
                
                # Format Letta style
                formatted = [f"[{res.get('score', 0):.2f}] {res.get('content')}" for res in (results or []) if isinstance(res, dict)]
                return {
                    "ok": True, 
                    "results": formatted if formatted else ["No archival memories found."],
                    "summary": f"Found {len(formatted)} artifacts."
                }
            except Exception as e:
                return {"ok": False, "error": f"Archival search failed: {e}"}

        return {"ok": False, "error": f"Unknown archival action: {action}"}
