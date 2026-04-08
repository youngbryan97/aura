import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.MemoryOps")


class MemoryOpsInput(BaseModel):
    action: str = Field(
        "remember",
        description="One of: remember, learn_fact, retrieve, or recall.",
    )
    key: Optional[str] = Field(None, description="Optional fact key for structured storage.")
    value: Optional[str] = Field(None, description="Optional fact value for structured storage.")
    content: Optional[str] = Field(None, description="Free-form text to store in long-term memory.")
    query: Optional[str] = Field(None, description="What to recall from memory.")


class MemoryOpsSkill(BaseSkill):
    name = "memory_ops"
    description = "Store and recall long-term facts, preferences, and important conversation details."
    input_model = MemoryOpsInput

    @staticmethod
    def _extract_objective(raw: Any, context: Dict[str, Any]) -> str:
        if isinstance(raw, dict):
            return str(
                raw.get("objective")
                or raw.get("query")
                or raw.get("content")
                or context.get("objective")
                or context.get("message")
                or ""
            ).strip()
        return str(context.get("objective") or context.get("message") or "").strip()

    @staticmethod
    def _normalize_fact_key(raw_key: str) -> Optional[str]:
        normalized = "_".join(token for token in str(raw_key or "").replace("'", "").split() if token)
        return normalized[:80] if normalized else None

    @classmethod
    def _derive_structured_fact(cls, text: str) -> tuple[Optional[str], Optional[str]]:
        import re

        cleaned = str(text or "").strip().strip(". ")
        patterns = (
            r"^(?:remember|store|save)(?:\s+for\s+(?:future\s+)?sessions?)?(?:\s+that)?\s+my\s+(.+?)\s+is\s+(.+)$",
            r"^my\s+(.+?)\s+is\s+(.+)$",
            r"^(?:remember|store|save)(?:\s+for\s+(?:future\s+)?sessions?)?(?:\s+that)?\s+(.+?)\s+is\s+(.+)$",
        )
        for pattern in patterns:
            match = re.match(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                key = cls._normalize_fact_key(match.group(1).strip().lower())
                raw_value = match.group(2).strip()
                if key and raw_value:
                    return key, raw_value[:400]
        return None, None

    @classmethod
    def _derive_query_key(cls, text: str) -> Optional[str]:
        import re

        cleaned = str(text or "").strip().strip("?!. ")
        patterns = (
            r"^(?:what\s+do\s+you\s+(?:remember|know)\s+about)\s+my\s+(.+)$",
            r"^(?:recall|retrieve)\s+my\s+(.+)$",
            r"^my\s+(.+)$",
        )
        for pattern in patterns:
            match = re.match(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                return cls._normalize_fact_key(match.group(1).strip().lower())
        return None

    def _coerce_input(self, raw: Any, context: Dict[str, Any]) -> MemoryOpsInput:
        if isinstance(raw, MemoryOpsInput):
            source = raw.model_dump(exclude_none=True)
        elif isinstance(raw, dict):
            source = dict(raw.get("params") or {}) if isinstance(raw.get("params"), dict) else dict(raw)
        else:
            source = {}

        objective = self._extract_objective(raw, context)
        action = str(source.get("action") or "").strip().lower()
        key = source.get("key")
        value = source.get("value")
        content = source.get("content")
        query = source.get("query")

        if not action:
            lowered = objective.lower()
            if any(marker in lowered for marker in ("what do you remember", "recall", "retrieve", "what do you know")):
                action = "recall"
            else:
                action = "remember"

        if not content and action in {"remember", "learn_fact"}:
            content = objective
        if not query and action in {"retrieve", "recall"}:
            query = objective or key
        if not key and action in {"retrieve", "recall"} and query:
            key = self._derive_query_key(query)

        if (not key or value is None) and content:
            derived_key, derived_value = self._derive_structured_fact(content)
            key = key or derived_key
            value = value if value is not None else derived_value

        return MemoryOpsInput(
            action=action,
            key=key,
            value=value,
            content=content,
            query=query,
        )

    @staticmethod
    async def _call_optional(method: Any, *args: Any, **kwargs: Any) -> Any:
        if method is None:
            return None
        if callable(method):
            result = method(*args, **kwargs)
            if hasattr(result, "__await__"):
                return await result
            return result
        return None

    async def execute(self, params: Any, context: Dict) -> Dict:
        request = self._coerce_input(params, context or {})
        action = request.action.lower().strip()

        memory_facade = context.get("memory_facade")
        memory_store = context.get("memory_store") or context.get("memory")
        semantic_memory = context.get("semantic_memory")

        if action in {"remember", "learn_fact"}:
            content = str(request.content or request.value or "").strip()
            key = str(request.key or "").strip()
            value = request.value
            metadata = {
                "source": "memory_ops",
                "origin": context.get("intent_source") or context.get("origin") or "memory_ops",
                "explicit_memory_request": True,
            }

            if key and value is not None and memory_store is not None:
                if hasattr(memory_store, "update_semantic_async"):
                    ok = await self._call_optional(memory_store.update_semantic_async, key, value)
                    if ok:
                        return {
                            "ok": True,
                            "summary": f"Stored fact: {key}.",
                            "result": {key: value},
                        }
                elif hasattr(memory_store, "update_semantic"):
                    ok = await self._call_optional(memory_store.update_semantic, key, value)
                    if ok:
                        return {
                            "ok": True,
                            "summary": f"Stored fact: {key}.",
                            "result": {key: value},
                        }

            if not content:
                return {"ok": False, "error": "No memory content provided."}

            if memory_facade and hasattr(memory_facade, "add_memory"):
                ok = await memory_facade.add_memory(content, metadata={**metadata, "key": key or None, "value": value})
                if ok:
                    return {
                        "ok": True,
                        "summary": "Stored that in long-term memory.",
                        "result": content,
                    }
                status = getattr(memory_facade, "_last_add_memory_status", {}) or {}
                reason = str(status.get("reason") or "memory_write_rejected").strip()
                return {"ok": False, "error": f"Memory write declined: {reason}."}

            if semantic_memory is not None:
                if hasattr(semantic_memory, "remember"):
                    await self._call_optional(semantic_memory.remember, content, metadata)
                    return {"ok": True, "summary": "Stored that in semantic memory.", "result": content}
                if hasattr(semantic_memory, "add_memory"):
                    await self._call_optional(semantic_memory.add_memory, content, metadata)
                    return {"ok": True, "summary": "Stored that in semantic memory.", "result": content}

            return {"ok": False, "error": "No writable memory backend is available."}

        if action in {"retrieve", "recall"}:
            key = str(request.key or "").strip()
            query = str(request.query or key or "").strip()

            if key and memory_store is not None:
                if hasattr(memory_store, "get_semantic_async"):
                    value = await self._call_optional(memory_store.get_semantic_async, key, None)
                    if value is not None:
                        return {"ok": True, "summary": f"Recalled {key}.", "result": value}
                elif hasattr(memory_store, "get_semantic"):
                    value = await self._call_optional(memory_store.get_semantic, key, None)
                    if value is not None:
                        return {"ok": True, "summary": f"Recalled {key}.", "result": value}

            if query and memory_facade and hasattr(memory_facade, "query_memory"):
                results = await memory_facade.query_memory(query, limit=3)
                if results:
                    return {
                        "ok": True,
                        "summary": f"Found {len(results)} memory match(es).",
                        "result": results,
                    }

            return {"ok": False, "error": f"No memory found for '{query or key}'."}

        return {"ok": False, "error": f"Unknown action: {request.action}"}
