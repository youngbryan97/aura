from core.runtime.atomic_writer import atomic_write_text
import logging
import os
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from pydantic import BaseModel, Field
from core.config import config
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.MemoryOps")


class MemoryOpsInput(BaseModel):
    action: str = Field(
        ...,
        description=(
            "Memory verb. Letta-style: 'core_append', 'core_replace', "
            "'archival_insert', 'archival_search'. High-level: 'remember', 'recall'."
        ),
    )
    block: Optional[str] = Field(None, description="Core Memory block name for core_* ops.")
    content: Optional[str] = Field(None, description="Data to append/insert/replace, or a 'remember' request.")
    old_content: Optional[str] = Field(None, description="Exact prior string to replace.")
    query: Optional[str] = Field(None, description="Search term or recall question.")
    key: Optional[str] = Field(None, description="Explicit key for remember/recall (skips natural-language derivation).")
    value: Optional[str] = Field(None, description="Explicit value for remember.")


@dataclass
class MemoryOpsRequest:
    """Internal coerced form of a memory-ops call.

    The wire format is ``MemoryOpsInput`` (Pydantic, validated). For
    high-level verbs ``remember``/``recall`` we collapse natural-language
    content into a structured (key, value) pair before reaching the store.
    """

    action: str
    key: Optional[str] = None
    value: Optional[str] = None
    query: Optional[str] = None
    content: Optional[str] = None
    block: Optional[str] = None
    old_content: Optional[str] = None
    raw: Optional[Any] = None


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
                atomic_write_text(path, f"// Core Memory Block: {block}\n", encoding="utf-8")

    # ── Natural-language → structured-fact derivation ──────────────────
    # Patterns are intentionally narrow so we only succeed when the user
    # explicitly stated an "X is Y" fact. Anything fuzzier should be left
    # unkeyed and routed to vector storage by the caller.
    _REMEMBER_PATTERNS = (
        re.compile(
            r"remember(?:\s+for\s+future\s+sessions)?(?:\s+that)?\s+(?:my\s+)?"
            r"(?P<key>.+?)\s+(?:is|=)\s+(?P<value>.+?)[\.!\?]?$",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:my\s+)?(?P<key>[a-z][a-z0-9 _-]+?)\s+(?:is|=)\s+(?P<value>.+?)[\.!\?]?$",
            re.IGNORECASE,
        ),
    )
    _RECALL_PATTERNS = (
        re.compile(
            r"what(?:\s+do\s+you)?\s+remember\s+about\s+(?:my\s+)?"
            r"(?P<key>.+?)[\?\.!]?$",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:tell\s+me|recall|what\s+is)\s+(?:my\s+)?(?P<key>.+?)[\?\.!]?$",
            re.IGNORECASE,
        ),
    )

    @staticmethod
    def _normalize_key(raw: str) -> str:
        cleaned = raw.strip().lower()
        cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
        return cleaned.strip("_")

    @staticmethod
    def _derive_structured_fact(content: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract a (key, value) pair from a natural-language remember request.

        Returns (None, None) if no clean fact can be derived.
        """
        if not content:
            return None, None
        text = content.strip()
        for pattern in MemoryOpsSkill._REMEMBER_PATTERNS:
            m = pattern.search(text)
            if m:
                key = MemoryOpsSkill._normalize_key(m.group("key"))
                value = m.group("value").strip().rstrip(".!?")
                if key and value:
                    return key, value
        return None, None

    @staticmethod
    def _derive_recall_key(query: str) -> Optional[str]:
        if not query:
            return None
        text = query.strip()
        for pattern in MemoryOpsSkill._RECALL_PATTERNS:
            m = pattern.search(text)
            if m:
                key = MemoryOpsSkill._normalize_key(m.group("key"))
                if key:
                    return key
        return None

    def _coerce_input(self, params: Any, context: Dict[str, Any]) -> MemoryOpsRequest:
        """Coerce wire input into the internal ``MemoryOpsRequest`` form.

        For ``remember``/``recall`` actions we derive (key, value) from the
        natural-language ``content`` / ``query`` if explicit fields weren't
        provided.
        """
        if isinstance(params, dict):
            params = MemoryOpsInput(**params)

        req = MemoryOpsRequest(
            action=params.action.lower(),
            key=params.key,
            value=params.value,
            query=params.query,
            content=params.content,
            block=params.block,
            old_content=params.old_content,
            raw=params,
        )

        if req.action == "remember" and req.key is None and req.content:
            k, v = self._derive_structured_fact(req.content)
            if k:
                req.key = k
                req.value = v if req.value is None else req.value

        if req.action == "recall" and req.key is None and req.query:
            k = self._derive_recall_key(req.query)
            if k:
                req.key = k

        return req

    async def execute(self, params: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            req = self._coerce_input(params, context)
        except Exception as e:
            return {"ok": False, "error": f"Invalid input: {e}"}

        action = req.action

        try:
            if action == "remember":
                return await self._execute_remember(req, context)
            if action == "recall":
                return await self._execute_recall(req, context)
            if action.startswith("core_"):
                return await self._execute_core_memory(req.raw, context, action)
            if action.startswith("archival_"):
                return await self._execute_archival_memory(req.raw, context, action)
            return {"ok": False, "error": f"Unknown memory action: {action}"}
        except Exception as e:
            logger.error("MemoryOps failed: %s", e)
            return {"ok": False, "error": str(e)}

    # ── High-level remember / recall ───────────────────────────────────

    async def _execute_remember(
        self, req: MemoryOpsRequest, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Store a derived (key, value) fact.

        Resolution order for the backing store:
          1. ``memory_facade.add_memory`` — gives the facade a chance to
             reject (e.g. constitutional / substrate gating). If it returns
             False the rejection is propagated, NOT bypassed via vector.
          2. ``memory_store.update_semantic_async(key, value)``
          3. ``semantic_memory.add_memory(text, metadata)``
        """
        if not req.key or req.value is None:
            return {
                "ok": False,
                "error": (
                    "Could not derive a structured fact from input. "
                    "Provide explicit key/value, or phrase as 'my <key> is <value>'."
                ),
            }

        facade = context.get("memory_facade")
        if facade is not None and hasattr(facade, "add_memory"):
            text = f"{req.key.replace('_', ' ')}: {req.value}"
            metadata = {
                "origin": context.get("intent_source", "user"),
                "explicit_memory_request": True,
                "derived_key": req.key,
            }
            res = facade.add_memory(text, metadata=metadata)
            if hasattr(res, "__await__"):
                res = await res
            if res is False:
                status = getattr(facade, "_last_add_memory_status", None) or {}
                reason = status.get("reason", "memory_facade_rejected")
                return {"ok": False, "error": f"memory_facade rejected: {reason}"}

        store = context.get("memory_store")
        if store is not None and hasattr(store, "update_semantic_async"):
            await store.update_semantic_async(req.key, req.value)
            return {"ok": True, "summary": f"Stored fact: {req.key}."}

        sem = context.get("semantic_memory")
        if sem is not None and hasattr(sem, "add_memory"):
            sem.add_memory(
                f"{req.key.replace('_', ' ')}: {req.value}",
                metadata={"derived_key": req.key},
            )
            return {"ok": True, "summary": f"Stored fact: {req.key}."}

        return {"ok": False, "error": "No memory backend wired in context."}

    async def _execute_recall(
        self, req: MemoryOpsRequest, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Retrieve a previously remembered fact by derived key."""
        if not req.key:
            return {
                "ok": False,
                "error": (
                    "Could not derive a key from the question. "
                    "Phrase as 'what do you remember about my <key>?'."
                ),
            }

        store = context.get("memory_store")
        if store is not None and hasattr(store, "get_semantic_async"):
            value = await store.get_semantic_async(req.key, None)
            return {"ok": True, "result": value, "key": req.key}

        return {"ok": False, "error": "No memory backend wired in context."}

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
