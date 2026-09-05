import hashlib
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.config import config
from core.container import ServiceContainer
from core.governance.will import ActionDomain
from core.runtime.action_executor import ActionExecutor
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
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
    block: str | None = Field(None, description="The Core Memory block name (e.g., 'persona', 'user') for core_* ops.")
    content: str | None = Field(None, description="Data to append, insert, or replace.")
    old_content: str | None = Field(None, description="Exact prior string to replace. Used only in 'core_replace'.")
    query: str | None = Field(None, description="Search term for 'archival_search'.")


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

    async def execute(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
        if isinstance(params, dict):
            try:
                params = MemoryOpsInput(**params)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('memory_ops', e)
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
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('memory_ops', e)
            logger.error("MemoryOps failed: %s", e)
            return {"ok": False, "error": str(e)}

    @classmethod
    def _normalize_action(cls, action: Any) -> str:
        lowered = str(action or "").strip().lower()
        return cls._ACTION_ALIASES.get(lowered, lowered)

    @staticmethod
    def _resolve_memory_facade(context: dict[str, Any]) -> Any:
        return context.get("memory_facade") or ServiceContainer.get("memory_facade", default=None)

    @staticmethod
    def _sha256_text(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _core_memory_effect(
        cls,
        block_path: Path,
        *,
        expected_sha256: str,
    ) -> dict[str, Any]:
        exists = block_path.exists()
        effect: dict[str, Any] = {
            "path": str(block_path),
            "exists": exists,
            "effect_verified": False,
        }
        if exists:
            sha256 = cls._file_sha256(block_path)
            effect.update({
                "sha256": sha256,
                "bytes": block_path.stat().st_size,
                "expected_sha256": expected_sha256,
                "effect_verified": sha256 == expected_sha256,
            })
        else:
            effect["expected_sha256"] = expected_sha256
        return effect

    @staticmethod
    def _archival_write_metadata(context: dict[str, Any]) -> dict[str, Any]:
        origin = str(context.get("origin") or context.get("source") or "").strip()
        explicit_request = bool(
            context.get("user_requested_action")
            or context.get("user_explicitly_authorized")
            or origin in {"user", "chat", "chat_api", "live_skill_api", "desktop_ui"}
        )
        metadata: dict[str, Any] = {
            "source": "archival_insert",
            "explicit_memory_request": explicit_request,
        }
        if origin:
            metadata["origin"] = origin
        return metadata

    @staticmethod
    def _archival_write_status(memory_facade: Any) -> dict[str, Any]:
        snapshot = getattr(memory_facade, "last_add_memory_status", None)
        if callable(snapshot):
            try:
                status = snapshot()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                status = None
            if isinstance(status, dict):
                return dict(status)
        status = getattr(memory_facade, "_last_add_memory_status", None)
        return dict(status) if isinstance(status, dict) else {}

    async def _execute_core_memory(self, params: MemoryOpsInput, context: dict[str, Any], action: str) -> dict[str, Any]:
        """RAM: Immediate context window blocks."""
        block = params.block or "user"
        if not block.isalnum() and "_" not in block:
            return {"ok": False, "error": "Invalid block name. Must be alphanumeric."}
            
        block_path = self.mem_fs_dir / f"{block}.txt"
        
        if action == "core_append":
            if not params.content:
                return {"ok": False, "error": "Missing 'content' to append."}
            current_content = ""
            if block_path.exists():
                with open(block_path, encoding="utf-8") as f:
                    current_content = f.read()
            new_content = current_content + params.content + "\n"
            result = await ActionExecutor.execute(
                domain=ActionDomain.FILE_WRITE,
                action_name="core_append",
                params={"path": str(block_path), "text": new_content},
                source="memory_ops",
            )
            if not result.get("ok"):
                return {
                    "ok": False,
                    "error": result.get("error", "core memory append failed"),
                    "block": block,
                }
            effect = self._core_memory_effect(
                block_path,
                expected_sha256=self._sha256_text(new_content),
            )
            return {
                "ok": bool(effect.get("effect_verified")),
                "summary": f"Appended to core memory block '{block}'.",
                "block": block,
                "criteria_results": {"core memory appended": bool(effect.get("effect_verified"))},
                **effect,
            }

        elif action == "core_replace":
            if not params.content or not params.old_content:
                return {"ok": False, "error": "Missing 'content' or 'old_content' for replacing."}
            
            with open(block_path, encoding="utf-8") as f:
                data = f.read()
                
            if params.old_content not in data:
                return {"ok": False, "error": f"Text to replace not found in block '{block}'."}
                
            new_data = data.replace(params.old_content, params.content)
            result = await ActionExecutor.execute(
                domain=ActionDomain.FILE_WRITE,
                action_name="core_replace",
                params={"path": str(block_path), "text": new_data},
                source="memory_ops",
            )
            if not result.get("ok"):
                return {
                    "ok": False,
                    "error": result.get("error", "core memory replace failed"),
                    "block": block,
                }
            effect = self._core_memory_effect(
                block_path,
                expected_sha256=self._sha256_text(new_data),
            )
            return {
                "ok": bool(effect.get("effect_verified")),
                "summary": f"Replaced content in core memory block '{block}'.",
                "block": block,
                "criteria_results": {"core memory replaced": bool(effect.get("effect_verified"))},
                **effect,
            }
            
        return {"ok": False, "error": f"Unknown core action: {action}"}

    async def _execute_archival_memory(self, params: MemoryOpsInput, context: dict[str, Any], action: str) -> dict[str, Any]:
        """Disk: Long-term archival Vector / DB storage."""
        memory_facade = self._resolve_memory_facade(context)
        if not memory_facade:
            return {"ok": False, "error": "Archival backend (memory_facade) is not wired to context."}

        if action == "archival_insert":
            if not params.content:
                return {"ok": False, "error": "Missing 'content' to archive."}
            
            try:
                write_result: Any
                if hasattr(memory_facade, "add_memory"):
                    write_result = memory_facade.add_memory(
                        params.content,
                        metadata=self._archival_write_metadata(context),
                    )
                    if hasattr(write_result, "__await__"):
                        write_result = await write_result
                elif hasattr(memory_facade, "update_semantic_async"):
                    write_result = await memory_facade.update_semantic_async(
                        "archival_" + str(len(params.content)),
                        params.content,
                    )
                else:
                    return {"ok": False, "error": "Facade missing insertion capability."}

                write_status = self._archival_write_status(memory_facade)
                if write_result is False or write_status.get("ok") is False:
                    reason = str(
                        write_status.get("reason")
                        or "archival backend rejected the write"
                    )
                    return {
                        "ok": False,
                        "error": f"Archival insertion failed: {reason}",
                        "storage_backend": str(write_status.get("backend") or ""),
                    }

                record_id = str(
                    write_status.get("record_id")
                    or getattr(write_result, "record_id", "")
                    or ""
                )
                receipt_id = str(
                    write_status.get("receipt_id")
                    or getattr(write_result, "receipt_id", "")
                    or ""
                )
                bytes_written = int(
                    write_status.get("bytes_written")
                    or getattr(write_result, "bytes_written", 0)
                    or 0
                )
                effect_verified = bool(record_id and receipt_id and bytes_written > 0)
                return {
                    "ok": True,
                    "status": "success_verified" if effect_verified else "success_unverified",
                    "summary": "Committed to archival storage.",
                    "storage_backend": str(
                        write_status.get("backend")
                        or write_status.get("reason")
                        or "legacy_memory_facade"
                    ),
                    "record_id": record_id,
                    "memory_receipt_id": receipt_id,
                    "bytes_written": bytes_written,
                    "content_sha256": self._sha256_text(params.content),
                    "effect_verified": effect_verified,
                    "criteria_results": {
                        "archival memory stored": effect_verified,
                    },
                }
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('memory_ops', e)
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
            except (OSError, ConnectionError, TimeoutError) as e:
                record_degradation('memory_ops', e)
                return {"ok": False, "error": f"Archival search failed: {e}"}

        return {"ok": False, "error": f"Unknown archival action: {action}"}
