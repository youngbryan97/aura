"""Refactored MemoryFacade — the central entry point for all long-term memory operations.
Ensures episodic and semantic sub-systems work in harmony.
"""
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import logging
import asyncio
import inspect
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from core.container import ServiceContainer

logger = logging.getLogger("Aura.Memory")

class MemoryFacade:
    """
    Unified entry point for episodic, semantic, and vector memories.
    Provides a simple API for the rest of the system to manage its continuity.
    """
    TECHNICAL_HINTS = (
        "code", "repo", "file", "function", "method", "class",
        "module", "symbol", "technical", "api", "endpoint", "schema",
    )
    FILE_METADATA_KEYS = (
        "file_path", "path", "source_file", "source_path",
        "repo_path", "relative_path", "target_file",
    )
    SIGNATURE_METADATA_KEYS = (
        "target_signature", "signature", "symbol", "function",
        "class_name", "method", "api_name",
    )
    FILE_REFERENCE_RE = re.compile(
        r"(?<![\w/.-])((?:[\w.-]+/)+[\w.-]+\.(?:py|tsx?|jsx?|json|md|ya?ml|toml|sh|go|rs|java|c|cpp|h))(?:[:#]\d+)?"
    )
    SYMBOL_PATTERNS = (
        re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)"),
        re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)"),
        re.compile(r"\b(?:function|method|module|symbol)\s+`?([A-Za-z_][A-Za-z0-9_]*)`?", re.IGNORECASE),
    )
    VERIFICATION_PRIORITY = {
        "verified_live": 0,
        "not_applicable": 1,
        "unverified": 2,
        "missing": 3,
        "stale": 4,
    }
    USER_FACING_SOURCES = frozenset({
        "user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external",
    })
    
    def __init__(self, orchestrator: Optional[Any] = None):
        """
        Initialize the facade.
        :param orchestrator: Optional reference to the orchestrator (legacy).
        """
        self._orchestrator = orchestrator
        # 1. Private backing fields
        self._episodic = None
        self._semantic = None 
        self._vector = None
        self._ledger = None
        self._graph = None
        self._short_term = None
        self._goals = None
        self._vault = None
        self._cold = None
        self._last_commit_time = None
        self._last_add_memory_status: Dict[str, Any] = {"ok": True, "reason": "not_attempted"}
        self._repo_root = Path(__file__).resolve().parents[2]

    def _refresh_subsystems(self) -> None:
        """Resolve subsystem handles from the container without requiring async boot."""
        self._episodic = ServiceContainer.get("episodic_memory", default=None)
        self._semantic = ServiceContainer.get("semantic_memory", default=None)
        self._vector = ServiceContainer.get("vector_memory", default=None)
        self._ledger = ServiceContainer.get("knowledge_ledger", default=None)
        self._graph = ServiceContainer.get("knowledge_graph", default=None)
        self._short_term = ServiceContainer.get("short_term_memory", default=None)
        self._goals = ServiceContainer.get("goal_memory", default=None)
        self._vault = ServiceContainer.get("blackhole_vault", default=None)
        self._cold = ServiceContainer.get("cold_store", default=None)
        
    def setup(self) -> None:
        """Legacy synchronous setup shim."""
        self._refresh_subsystems()
        logger.debug("MemoryFacade.setup() resolved subsystems synchronously.")
        
    async def on_start_async(self) -> None:
        """Lifecycle hook for async initialization."""
        logger.info("🧠 MemoryFacade: Initializing memory systems (async)...")
        self._refresh_subsystems()
        
        # Verify connectivity (non-blocking)
        logger.info("✓ MemoryFacade: Subsystems online.")
        
        logger.info("MemoryFacade setup complete (E:%s S:%s V:%s L:%s G:%s ST:%s)",
                    bool(self._episodic), bool(self._semantic), bool(self._vector),
                    bool(self._ledger), bool(self._graph), bool(self._short_term))

    @property
    def episodic(self): return self._episodic
    @property
    def vector(self): return self._vector
    @property
    def semantic(self): return self._semantic
    @property
    def ledger(self): return self._ledger
    @property
    def graph(self): return self._graph
    @property
    def short_term(self): return self._short_term
    @property
    def goals(self): return self._goals
    @property
    def vault(self): return self._vault
    @property
    def cold(self): return self._cold

    def _normalize_memory_result(
        self,
        *,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: str = "",
        score: Optional[float] = None,
    ) -> Dict[str, Any]:
        payload = {
            "id": memory_id,
            "text": content,
            "content": content,
            "metadata": dict(metadata or {}),
        }
        if score is not None:
            payload["score"] = score
        return payload

    def _extract_candidate_path(self, metadata: Dict[str, Any], content: str) -> Optional[str]:
        for key in self.FILE_METADATA_KEYS:
            raw_value = str(metadata.get(key) or "").strip()
            if raw_value:
                return raw_value
        match = self.FILE_REFERENCE_RE.search(content or "")
        if match:
            return match.group(1)
        return None

    @classmethod
    def _normalize_source_label(cls, raw: Any) -> str:
        return str(raw or "").strip().lower().replace("-", "_")

    @classmethod
    def _is_user_facing_source(cls, raw: Any) -> bool:
        normalized = cls._normalize_source_label(raw)
        if not normalized:
            return False
        if normalized in cls.USER_FACING_SOURCES:
            return True
        tokens = {token for token in normalized.split("_") if token}
        return bool(tokens & cls.USER_FACING_SOURCES)

    def _resolve_memory_write_source(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        fallback: str = "memory_facade",
    ) -> str:
        payload = dict(metadata or {})
        for key in ("intent_source", "origin", "request_origin", "source"):
            candidate = self._normalize_source_label(payload.get(key))
            if self._is_user_facing_source(candidate):
                return candidate or "user"
        return fallback

    def _should_degrade_add_memory_block(
        self,
        reason: Any,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        source: str,
    ) -> bool:
        """Allow legacy local writes to degrade open when governance is unavailable.

        The compatibility ``add_memory()`` API is still used by low-level tests and
        non-user-facing plumbing. Those callers should not fail closed purely
        because a partial runtime left strict governance services registered while
        their prerequisites (for example the self-model) are still unavailable.
        Explicit/user-facing memory writes still fail closed.
        """
        if self._orchestrator is not None:
            return False

        payload = dict(metadata or {})
        if payload.get("explicit_memory_request"):
            return False
        if self._is_user_facing_source(source):
            return False

        normalized_reason = str(reason or "").strip().lower()
        degraded_reasons = (
            "self_model_required",
            "executive_core_required",
            "authority_gateway_required",
            "authority_gateway_unavailable",
            "constitutional_gate_unavailable",
        )
        return any(normalized_reason.startswith(prefix) for prefix in degraded_reasons)

    def _should_store_semantic_interaction(
        self,
        *,
        metadata: Optional[Dict[str, Any]],
        success: bool,
        importance: float,
        action: str,
    ) -> bool:
        payload = dict(metadata or {})
        if self._is_user_facing_source(self._resolve_memory_write_source(payload, fallback="")):
            return True
        if not success or importance >= 0.75:
            return True
        if float(payload.get("memory_salience", 0.0) or 0.0) >= 0.55:
            return True
        return str(action or "").startswith(("conversation", "execute_tool("))

    @staticmethod
    def _build_semantic_interaction_text(
        *,
        context: str,
        action: str,
        outcome: str,
        metadata: Optional[Dict[str, Any]],
    ) -> str:
        payload = dict(metadata or {})
        objective = str(payload.get("objective") or context or "").strip()
        action_text = str(action or "").strip()
        outcome_text = str(outcome or "").strip()

        if action_text.startswith("execute_tool(") and action_text.endswith(")"):
            tool_name = action_text[len("execute_tool("):-1]
            return f"Objective: {objective}\nTool: {tool_name}\nOutcome: {outcome_text[:900]}".strip()

        if action_text.startswith("conversation"):
            return f"User: {objective}\nAura: {outcome_text[:900]}".strip()

        return (
            f"Context: {objective}\n"
            f"Action: {action_text}\n"
            f"Outcome: {outcome_text[:900]}"
        ).strip()

    def _resolve_candidate_path(self, raw_path: Optional[str]) -> Optional[Path]:
        cleaned = str(raw_path or "").strip().strip("`\"'")
        if not cleaned:
            return None
        if ":" in cleaned and not cleaned.startswith(("/", "./", "../")):
            base, suffix = cleaned.rsplit(":", 1)
            if suffix.isdigit():
                cleaned = base
        candidate = Path(cleaned).expanduser()
        if candidate.is_absolute():
            return candidate
        return (self._repo_root / candidate).resolve()

    def _extract_candidate_signature(self, metadata: Dict[str, Any], content: str) -> Optional[str]:
        for key in self.SIGNATURE_METADATA_KEYS:
            raw_value = str(metadata.get(key) or "").strip()
            if raw_value:
                return raw_value
        for pattern in self.SYMBOL_PATTERNS:
            match = pattern.search(content or "")
            if match:
                return match.group(1)
        return None

    def _looks_technical_memory(self, metadata: Dict[str, Any], content: str) -> bool:
        if self._extract_candidate_path(metadata, content):
            return True
        meta_blob = " ".join(
            str(metadata.get(key) or "")
            for key in ("type", "source", "category", "domain", "kind", "memory_type")
        )
        combined = f"{meta_blob} {content or ''}".lower()
        return any(hint in combined for hint in self.TECHNICAL_HINTS)

    async def _verify_memory_result(self, item: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(item)
        metadata = dict(normalized.get("metadata") or {})
        content = str(normalized.get("content") or normalized.get("text") or "").strip()

        verification_state = "not_applicable"
        verification_evidence = ""

        if self._looks_technical_memory(metadata, content):
            verification_state = "unverified"
            raw_path = self._extract_candidate_path(metadata, content)
            candidate_path = self._resolve_candidate_path(raw_path)
            if candidate_path is not None:
                metadata["resolved_path"] = str(candidate_path)
                if await asyncio.to_thread(candidate_path.exists):
                    signature = self._extract_candidate_signature(metadata, content)
                    if signature:
                        try:
                            live_content = await asyncio.to_thread(
                                candidate_path.read_text,
                                encoding="utf-8",
                                errors="ignore",
                            )
                        except Exception:
                            live_content = ""

                        if signature in live_content:
                            verification_state = "verified_live"
                            verification_evidence = f"matched '{signature}' in {candidate_path.name}"
                        else:
                            verification_state = "stale"
                            verification_evidence = f"file exists but '{signature}' was not found"
                    else:
                        verification_state = "verified_live"
                        verification_evidence = f"live file exists: {candidate_path.name}"
                else:
                    verification_state = "missing"
                    verification_evidence = f"live file missing: {candidate_path}"

        metadata["verification_state"] = verification_state
        if verification_evidence:
            metadata["verification_evidence"] = verification_evidence
        normalized["metadata"] = metadata
        return normalized

    def _parse_memory_query(self, query: str) -> tuple[str, Optional[str], str]:
        raw = str(query or "").strip()
        if ":" not in raw:
            return "", None, raw
        key, value = raw.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in {"type", "tag", "source"} and value:
            return key, value, ""
        return "", None, raw

    def _filter_vector_records(
        self,
        records: List[Dict[str, Any]],
        *,
        filter_key: str,
        filter_value: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if not records:
            return normalized

        for record in records:
            metadata = dict(record.get("metadata") or {})
            if filter_key and filter_value is not None:
                if str(metadata.get(filter_key, "")).strip() != filter_value:
                    continue
            content = str(record.get("content") or record.get("text") or "")
            normalized.append(
                self._normalize_memory_result(
                    content=content,
                    metadata=metadata,
                    memory_id=str(record.get("id", "") or ""),
                    score=record.get("score"),
                )
            )

        normalized.sort(
            key=lambda item: float(item["metadata"].get("timestamp", 0.0) or 0.0),
            reverse=True,
        )
        return normalized[:limit]

    def _query_vector_memory_sync(
        self,
        query: str,
        *,
        filter_key: str,
        filter_value: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        if not self.vector:
            return []

        if filter_key and hasattr(self.vector, "_store"):
            return self._filter_vector_records(
                list(getattr(self.vector, "_store", []) or []),
                filter_key=filter_key,
                filter_value=filter_value,
                limit=limit,
            )

        if filter_key and hasattr(self.vector, "_collection"):
            try:
                results = self.vector._collection.get(include=["documents", "metadatas"])
                docs = list(results.get("documents", []) or [])
                metas = list(results.get("metadatas", []) or [])
                ids = list(results.get("ids", []) or [])
                records = [
                    {
                        "id": ids[idx] if idx < len(ids) else "",
                        "content": docs[idx] if idx < len(docs) else "",
                        "metadata": metas[idx] if idx < len(metas) else {},
                    }
                    for idx in range(max(len(docs), len(metas), len(ids)))
                ]
                return self._filter_vector_records(
                    records,
                    filter_key=filter_key,
                    filter_value=filter_value,
                    limit=limit,
                )
            except Exception as e:
                record_degradation('memory_facade', e)
                logger.debug("Vector collection metadata query failed: %s", e)

        if hasattr(self.vector, "search_similar"):
            results = self.vector.search_similar(query or filter_value or "", limit=limit)
        elif hasattr(self.vector, "search"):
            results = self.vector.search(query or filter_value or "", limit=limit)
        else:
            return []

        return self._filter_vector_records(
            list(results or []),
            filter_key=filter_key,
            filter_value=filter_value,
            limit=limit,
        )

    @staticmethod
    async def _call_maybe_async(method: Any, *args: Any, **kwargs: Any) -> Any:
        if method is None:
            return None
        if asyncio.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        result = await asyncio.to_thread(method, *args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def commit_interaction(self,
                                 context: str,
                                 action: str,
                                 outcome: str,
                                 success: bool,
                                 emotional_valence: float = 0.0,
                                 importance: float = 0.5,
                                 metadata: Optional[Dict[str, Any]] = None):
        """Unified commit for an interaction across all relevant systems."""
        resolved_source = self._resolve_memory_write_source(metadata)
        governance_decision = None
        try:
            from core.container import ServiceContainer
            from core.constitution import get_constitutional_core, unpack_governance_result

            approved, reason, governance_decision = unpack_governance_result(
                await get_constitutional_core(self._orchestrator).approve_memory_write(
                    memory_type="interaction_commit",
                    content=f"{context[:160]} -> {action[:80]} -> {outcome[:160]}",
                    source=resolved_source,
                    importance=max(0.0, min(1.0, float(importance or 0.0))),
                    metadata={"success": bool(success), **dict(metadata or {})},
                    return_decision=True,
                )
            )
            if not approved:
                logger.info("MemoryFacade: deferring interaction commit: %s", reason)
                return None
        except Exception as exc:
            record_degradation('memory_facade', exc)
            logger.debug("MemoryFacade constitutional gate skipped: %s", exc)
            runtime_live = bool(
                getattr(ServiceContainer, "_registration_locked", False)
                or ServiceContainer.has("executive_core")
                or ServiceContainer.has("aura_kernel")
                or ServiceContainer.has("kernel_interface")
            )
            if runtime_live:
                logger.warning("🚫 MemoryFacade interaction commit blocked: constitutional gate unavailable")
                return None

        self._last_commit_time = datetime.now()
        metadata = metadata or {}

        async def _commit_interaction_effects() -> Optional[Any]:
            if os.environ.get("AURA_STRICT_RUNTIME") == "1":
                from core.memory.memory_write_gateway import get_memory_write_gateway
                from core.runtime.gateways import MemoryWriteRequest
                try:
                    gw = get_memory_write_gateway()
                    payload = {"context": context, "action": action, "outcome": outcome, "success": success, "emotional_valence": emotional_valence, "importance": importance, **(metadata or {})}
                    await gw.write(MemoryWriteRequest(content=f"Interaction: {context} -> {action} -> {outcome}", metadata=payload, cause="memory_facade.commit_interaction"))
                    return "gateway-receipt"
                except PermissionError as e:
                    raise RuntimeError(f"Strict Runtime: memory write blocked: {e}") from e

            # 1. Record as Episode
            episode_id = None
            if self.episodic:
                try:
                    episode_id = await self.episodic.record_episode_async(
                        context=context,
                        action=action,
                        outcome=outcome,
                        success=success,
                        emotional_valence=emotional_valence,
                        importance=importance,
                        source=resolved_source,
                        metadata=metadata,
                    )
                except Exception as e:
                    record_degradation('memory_facade', e)
                    logger.error("Failed to record episode: %s", e)

            semantic_target = self.semantic if self.semantic is not None else self.vector
            semantic_write_ok = False
            if semantic_target and self._should_store_semantic_interaction(
                metadata=metadata,
                success=success,
                importance=importance,
                action=action,
            ):
                semantic_text = self._build_semantic_interaction_text(
                    context=context,
                    action=action,
                    outcome=outcome,
                    metadata=metadata,
                )
                semantic_metadata = {
                    "episode_id": episode_id,
                    "success": success,
                    "importance": importance,
                    "memory_type": "interaction_semantic",
                    **dict(metadata or {}),
                }
                try:
                    if hasattr(semantic_target, "remember"):
                        await self._call_maybe_async(semantic_target.remember, semantic_text, semantic_metadata)
                    elif hasattr(semantic_target, "add_memory"):
                        await self._call_maybe_async(semantic_target.add_memory, semantic_text, semantic_metadata)
                    elif hasattr(semantic_target, "index"):
                        await self._call_maybe_async(semantic_target.index, semantic_text, semantic_metadata)
                    semantic_write_ok = True
                except Exception as e:
                    record_degradation('memory_facade', e)
                    logger.error("Failed to update semantic memory: %s", e)

            # 2. Update Vector Memory if important
            if self.vector and (importance > 0.7 or success is False) and (
                self.vector is not semantic_target or not semantic_write_ok
            ):
                try:
                    await self._call_maybe_async(
                        self.vector.add_memory,
                        content=f"Interaction: {context} -> {action} -> {outcome}",
                        metadata={
                            "episode_id": episode_id,
                            "success": success,
                            "importance": importance,
                            **metadata
                        }
                    )
                except Exception as e:
                    record_degradation('memory_facade', e)
                    logger.error("Failed to update vector memory: %s", e)

            if self.ledger and hasattr(self.ledger, "log_interaction"):
                try:
                    await self._call_maybe_async(
                        self.ledger.log_interaction,
                        action,
                        outcome,
                        success,
                    )
                except Exception as e:
                    record_degradation('memory_facade', e)
                    logger.debug("Failed to update knowledge ledger: %s", e)

            return episode_id

        if governance_decision is not None:
            from core.governance_context import governed_scope

            async with governed_scope(governance_decision):
                return await _commit_interaction_effects()
        return await _commit_interaction_effects()

    async def get_hot_memory(self, limit: int = 5) -> Dict[str, Any]:
        """Retrieve recent interaction history and context for active thought."""
        hot = {
            "recent_episodes": [],
            "current_goals": [],
            "short_term": {}
        }
        
        if self.episodic:
            try:
                # Fix: Correct method name and use async
                recent = await self.episodic.recall_recent_async(limit=limit)
                hot["recent_episodes"] = recent
            except Exception as e:
                record_degradation('memory_facade', e)
                logger.debug("Failed to get recent episodes: %s", e)

        if self.goals:
            try:
                hot["current_goals"] = await self.goals.get_active_goals_async()
            except Exception as e:
                record_degradation('memory_facade', e)
                logger.debug("Failed to get active goals: %s", e)

        if self.short_term:
            hot["short_term"] = self.short_term.get_context()

        return hot

    async def search(self, query: str, limit: int = 5) -> List[Any]:
        """Search across all memory systems for relevance."""
        results: List[Dict[str, Any]] = []
        seen: set[str] = set()

        def _append(item: Any) -> None:
            if isinstance(item, dict):
                content = str(item.get("content") or item.get("text") or "").strip()
                metadata = dict(item.get("metadata") or {})
                normalized = self._normalize_memory_result(
                    content=content,
                    metadata=metadata,
                    memory_id=str(item.get("id", "") or ""),
                    score=item.get("score"),
                )
            else:
                normalized = self._normalize_memory_result(content=str(item or "").strip())

            key = f"{normalized.get('id', '')}::{normalized['content']}".lower()
            if not normalized["content"] or key in seen:
                return
            seen.add(key)
            results.append(normalized)
        
        # 1. Vector stores
        if self.vector:
            try:
                search_method = None
                if hasattr(self.vector, "search_memories"):
                    search_method = self.vector.search_memories
                elif hasattr(self.vector, "search_similar"):
                    search_method = self.vector.search_similar
                elif hasattr(self.vector, "search"):
                    search_method = self.vector.search

                if search_method is not None:
                    for item in list(await self._call_maybe_async(search_method, query, limit=limit) or []):
                        _append(item)
            except Exception as e:
                record_degradation('memory_facade', e)
                logger.debug("Vector search failed: %s", e)

        # 2. Semantic Graph
        if self.graph:
            try:
                search_method = self.graph.search_knowledge if hasattr(self.graph, "search_knowledge") else None
                if search_method is not None:
                    for item in list(await self._call_maybe_async(search_method, query, limit=limit) or []):
                        _append(item)
            except Exception as e:
                record_degradation('memory_facade', e)
                logger.debug("Graph search failed: %s", e)

        verified_results: List[Dict[str, Any]] = []
        for order, item in enumerate(results):
            try:
                normalized = await self._verify_memory_result(item)
            except Exception as e:
                record_degradation('memory_facade', e)
                logger.debug("Memory live verification failed: %s", e)
                normalized = dict(item)
                metadata = dict(normalized.get("metadata") or {})
                metadata.setdefault("verification_state", "unverified")
                normalized["metadata"] = metadata
            normalized["_retrieval_order"] = order
            verified_results.append(normalized)

        verified_results.sort(
            key=lambda item: (
                self.VERIFICATION_PRIORITY.get(
                    str(item.get("metadata", {}).get("verification_state", "not_applicable")),
                    2,
                ),
                item.get("_retrieval_order", 0),
            )
        )
        for item in verified_results:
            item.pop("_retrieval_order", None)
        return verified_results[:limit]

    async def add_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Compatibility API for legacy callers expecting async long-term memory writes."""
        payload = dict(metadata or {})
        # Provenance envelope: every memory write gets stamped with
        # source / confidence / identity_relevant / contested so downstream
        # readers can distinguish memory from inference / fantasy.
        try:
            from core.memory.provenance import wrap as _provenance_wrap
            _stamped = _provenance_wrap(
                text,
                source=str(payload.get("provenance_source") or payload.get("source") or "self_inferred"),
                confidence=payload.get("confidence"),
                identity_relevant=bool(payload.get("identity_relevant", False)),
                contested=bool(payload.get("contested", False)),
            )
            payload["provenance"] = {
                "record_id": _stamped.provenance.record_id,
                "when_created": _stamped.provenance.when_created,
                "source": _stamped.provenance.source,
                "confidence": _stamped.provenance.confidence,
                "contested": _stamped.provenance.contested,
                "identity_relevant": _stamped.provenance.identity_relevant,
                "schema_version": _stamped.provenance.schema_version,
            }
        except Exception as _prov_exc:
            record_degradation('memory_facade', _prov_exc)
            logger.debug("provenance stamp skipped: %s", _prov_exc)
        resolved_source = self._resolve_memory_write_source(payload)
        self._last_add_memory_status = {"ok": False, "reason": "pending"}
        governance_decision = None

        try:
            from core.container import ServiceContainer
            from core.constitution import get_constitutional_core, unpack_governance_result

            approved, reason, governance_decision = unpack_governance_result(
                await get_constitutional_core(self._orchestrator).approve_memory_write(
                    memory_type="facade_add_memory",
                    content=text,
                    source=resolved_source,
                    importance=max(0.0, min(1.0, float(payload.get("importance", 0.5) or 0.5))),
                    metadata=payload,
                    return_decision=True,
                )
            )
            if not approved:
                if self._should_degrade_add_memory_block(reason, payload, source=resolved_source):
                    logger.info(
                        "MemoryFacade add_memory: degrading governance block for legacy local write (%s).",
                        reason,
                    )
                    governance_decision = None
                else:
                    self._last_add_memory_status = {"ok": False, "reason": str(reason or "write_rejected")}
                    logger.warning("🚫 MemoryFacade add_memory blocked: %s", reason)
                    return False
        except Exception as exc:
            record_degradation('memory_facade', exc)
            logger.debug("MemoryFacade add_memory constitutional gate skipped: %s", exc)
            runtime_live = bool(
                getattr(ServiceContainer, "_registration_locked", False)
                or ServiceContainer.has("executive_core")
                or ServiceContainer.has("aura_kernel")
                or ServiceContainer.has("kernel_interface")
            )
            if runtime_live and not self._should_degrade_add_memory_block(
                "constitutional_gate_unavailable",
                payload,
                source=resolved_source,
            ):
                self._last_add_memory_status = {"ok": False, "reason": "constitutional_gate_unavailable"}
                logger.warning("🚫 MemoryFacade add_memory blocked: constitutional gate unavailable")
                return False

        async def _perform_add_memory() -> bool:
            if os.environ.get("AURA_STRICT_RUNTIME") == "1":
                from core.memory.memory_write_gateway import get_memory_write_gateway
                from core.runtime.gateways import MemoryWriteRequest
                try:
                    gw = get_memory_write_gateway()
                    await gw.write(MemoryWriteRequest(content=text, metadata=payload, cause="memory_facade.add_memory"))
                    self._last_add_memory_status = {"ok": True, "reason": "stored_via_gateway"}
                    return True
                except PermissionError as e:
                    self._last_add_memory_status = {"ok": False, "reason": f"gateway_error:{type(e).__name__}"}
                    raise RuntimeError(f"Strict Runtime: memory write blocked: {e}") from e

            if self.vector and hasattr(self.vector, "add_memory"):
                try:
                    raw_result = await asyncio.to_thread(self.vector.add_memory, text, payload)
                    stored = True if raw_result is None else bool(raw_result)
                    self._last_add_memory_status = {"ok": stored, "reason": "stored_via_vector" if stored else "vector_backend_returned_false"}
                    return stored
                except Exception as e:
                    record_degradation('memory_facade', e)
                    self._last_add_memory_status = {"ok": False, "reason": f"vector_backend_error:{type(e).__name__}"}
                    logger.error("MemoryFacade.add_memory via vector failed: %s", e)

            if self.semantic:
                try:
                    if hasattr(self.semantic, "remember"):
                        if asyncio.iscoroutinefunction(self.semantic.remember):
                            await self.semantic.remember(text, payload)
                        else:
                            await asyncio.to_thread(self.semantic.remember, text, payload)
                        self._last_add_memory_status = {"ok": True, "reason": "stored_via_semantic.remember"}
                        return True
                    if hasattr(self.semantic, "add_memory"):
                        await asyncio.to_thread(self.semantic.add_memory, text, payload)
                        self._last_add_memory_status = {"ok": True, "reason": "stored_via_semantic.add_memory"}
                        return True
                except Exception as e:
                    record_degradation('memory_facade', e)
                    self._last_add_memory_status = {"ok": False, "reason": f"semantic_backend_error:{type(e).__name__}"}
                    logger.error("MemoryFacade.add_memory via semantic failed: %s", e)

            if self.vault and hasattr(self.vault, "add_memory"):
                try:
                    raw_result = await asyncio.to_thread(self.vault.add_memory, text, payload)
                    stored = True if raw_result is None else bool(raw_result)
                    self._last_add_memory_status = {"ok": stored, "reason": "stored_via_vault" if stored else "vault_backend_returned_false"}
                    return stored
                except Exception as e:
                    record_degradation('memory_facade', e)
                    self._last_add_memory_status = {"ok": False, "reason": f"vault_backend_error:{type(e).__name__}"}
                    logger.error("MemoryFacade.add_memory via vault failed: %s", e)

            self._last_add_memory_status = {"ok": False, "reason": "no_writable_memory_backend"}
            return False

        if governance_decision is not None:
            from core.governance_context import governed_scope

            async with governed_scope(governance_decision):
                return await _perform_add_memory()
        return await _perform_add_memory()

    async def query_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Compatibility API for legacy narrative/semantic recall callers."""
        filter_key, filter_value, semantic_query = self._parse_memory_query(query)

        if self.vector:
            try:
                results = await asyncio.to_thread(
                    self._query_vector_memory_sync,
                    semantic_query or query,
                    filter_key=filter_key,
                    filter_value=filter_value,
                    limit=limit,
                )
                if results:
                    return results
            except Exception as e:
                record_degradation('memory_facade', e)
                logger.error("MemoryFacade.query_memory via vector failed: %s", e)

        if self.semantic and hasattr(self.semantic, "search_memories"):
            try:
                raw_results = await asyncio.to_thread(
                    self.semantic.search_memories,
                    semantic_query or query,
                    limit,
                )
                return self._filter_vector_records(
                    [
                        {
                            "id": str(item.get("id", "") or ""),
                            "content": str(item.get("content") or item.get("text") or ""),
                            "metadata": dict(item.get("metadata") or {}),
                            "score": item.get("score"),
                        }
                        for item in list(raw_results or [])
                    ],
                    filter_key=filter_key,
                    filter_value=filter_value,
                    limit=limit,
                )
            except Exception as e:
                record_degradation('memory_facade', e)
                logger.error("MemoryFacade.query_memory via semantic failed: %s", e)

        return []

    def log_event(self, event: Any) -> bool:
        """Lightweight event logger (Sync wrapper for fire-and-forget)."""
        if self.episodic:
            try:
                # Use create_task for non-blocking log
                get_task_tracker().create_task(self.episodic.log_event_async(event))
                return True
            except Exception as e:
                record_degradation('memory_facade', e)
                logger.debug("Sync log_event failed: %s", e)
        return False

    async def wipe(self, verify: bool = False) -> bool:
        """Danger: Clear all memories."""
        if not verify:
            return False
            
        logger.warning("☣️ Wiping ALL memories...")
        
        tasks = []
        if self._episodic: tasks.append(self._episodic.wipe())
        if self._vector: tasks.append(self._vector.wipe())
        if self._graph: tasks.append(self._graph.wipe())
        
        if tasks:
            await asyncio.gather(*tasks)
            
        logger.info("✓ Memory wipe complete.")
        return True

    def get_status(self) -> Dict[str, Any]:
        """Return a compact sync-friendly status payload for health checks and tests."""
        return {
            "episodic": self._episodic is not None,
            "semantic": self._semantic is not None,
            "vector": self._vector is not None,
            "ledger": self._ledger is not None,
            "graph": self._graph is not None,
            "short_term": self._short_term is not None,
            "goals": self._goals is not None,
            "vault": self._vault is not None,
            "cold": self._cold is not None,
            "last_commit": self._last_commit_time.isoformat() if self._last_commit_time else None,
        }

    def __repr__(self):
        return f"<MemoryFacade(E:{bool(self._episodic)} S:{bool(self._semantic)})>"
