"""Refactored MemoryFacade — the central entry point for long-term memory operations.
Ensures episodic and semantic sub-systems work in harmony.

Welfare integration stamps trace evidence into memory metadata and defers
high-integrity-risk writes before persistence.
"""
import asyncio
import inspect
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.Memory")

# ── Probe-harness memory hygiene ─────────────────────────────────────────────
# Endurance/soak probes drive the REAL /api/chat lane on purpose (that is the
# claim they test), so their turns look exactly like lived conversation. They
# must never persist into long-term memory or resurface in recall: probe turns
# like "(turn 169) In a few sentences, how does a refrigerator move heat?"
# re-emerging mid-chat produced visible thread-jumping drift for the user.
_PROBE_HARNESS_TURN_RE = re.compile(r"\(turn\s+\d+\)\s")
_PROBE_SESSION_ID_RE = re.compile(r"^(endurance|soak|probe|bench)[-_]", re.IGNORECASE)


def _probe_harness_reason(text: Any, metadata: Any) -> str:
    """Return why this content is probe-harness material, or '' if it is not."""
    payload = metadata if isinstance(metadata, dict) else {}
    if bool(payload.get("ephemeral_probe_session")):
        return "ephemeral_probe_session"
    session_id = str(
        payload.get("session_id") or payload.get("chat_session_id") or ""
    ).strip()
    if session_id and _PROBE_SESSION_ID_RE.match(session_id):
        return "probe_session_id"
    if _PROBE_HARNESS_TURN_RE.search(str(text or "")[:400]):
        return "harness_turn_pattern"
    return ""

def _writes_go_through_the_gateway() -> bool:
    """Whether a memory write goes through the governed gateway. Default yes.

    This read AURA_STRICT_RUNTIME with a default of "1", and every other
    reader of that variable in the tree treats its absence as off. So one flag
    meant two opposite things: eleven modules asked "should I fail hard on an
    anomaly", and this one asked "should the write be governed" — with the
    answer inverted. Setting AURA_STRICT_RUNTIME=0 to relax the first quietly
    rerouted memory writes around the gateway and its permission check.

    Its own name, defaulting to what this module already did, so the coupling
    goes and the behaviour does not.
    """
    said = os.environ.get("AURA_MEMORY_WRITES_UNGOVERNED", "").strip()
    return said not in ("1", "true", "yes")


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
        "admin",
        "api",
        "chat",
        "chat_api",
        "desktop",
        "desktop_task",
        "desktop_ui",
        "direct",
        "external",
        "frontend",
        "gui",
        "interface",
        "live_chat",
        "session_memory_pin",
        "ui",
        "user",
        "voice",
        "voice_bridge",
        "voice_input",
        "websocket",
        "ws",
    })
    PRINCIPAL_SCOPED_MEMORY_TYPES = frozenset({
        "conversation",
        "user_input",
    })
    PRINCIPAL_SCOPED_MEMORY_SOURCES = frozenset({
        "chat_turn_logger",
        "conversation",
        "conversation_identity",
        "session_memory_pin",
    })
    
    # Significance markers for bonding/relational conversations
    RELATIONAL_KEYWORDS = {
        # Bonding/relationship markers
        "bonding", "understand", "know each other", "friend", "trust", "care about",
        "connection", "relationship", "mutual", "together", "promise", "co-pilot",
        "building", "shared", "we have", "understand me", "know me",
        # Emotional depth
        "secret", "real", "honest", "genuine", "authentically", "truly",
        "heartfelt", "sincere", "vulnerable", "deeper", "intimate",
        # Joint endeavors/promises
        "travel", "ship", "adventure", "explore", "discover", "future",
        "journey", "quest", "mission", "starship",
        # Personal significance
        "dream", "wish", "hope", "aspiration", "goal", "wish for",
        "meaningful", "significant", "important", "matters",
    }
    
    def __init__(self, orchestrator: Any | None = None):
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
        self._last_add_memory_status: dict[str, Any] = {"ok": True, "reason": "not_attempted"}
        self._repo_root = Path(__file__).resolve().parents[2]

    def _refresh_subsystems(self) -> None:
        # Publish the memory health fragment for exactly as long as a facade
        # exists. Called rather than relying on an import side effect: a module
        # is imported once per process, so a registration that only happens at
        # import cannot be re-established after a reset or a hot reload, and a
        # health fragment that silently stops publishing is the failure this
        # register was built to remove.
        from core.memory.memory_inventory import register_memory_health_fragment

        register_memory_health_fragment()

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

    def is_ready(self) -> bool:
        """Deep liveness probe for the runtime health contract."""
        self._refresh_subsystems()
        return any(
            self._memory_backend_ready(
                subsystem,
                read_methods=read_methods,
                write_methods=write_methods,
            )
            for subsystem, read_methods, write_methods in (
                (
                    self._episodic,
                    ("recall_recent_async", "recall_recent", "search"),
                    ("record_episode_async", "record_episode", "add_memory"),
                ),
                (
                    self._semantic,
                    ("search_memories", "search", "get"),
                    ("remember", "add_memory", "index"),
                ),
                (
                    self._vector,
                    ("search_memories", "search_similar", "search"),
                    ("add_memory", "remember", "index"),
                ),
                (
                    self._graph,
                    ("search_knowledge", "search"),
                    ("update_belief", "upsert_node", "add_memory"),
                ),
                (
                    self._vault,
                    ("search", "get_recent", "get"),
                    ("add_memory", "remember"),
                ),
                (
                    self._cold,
                    ("search", "retrieve", "get"),
                    ("add_memory", "store", "write"),
                ),
            )
        )

    def last_add_memory_status(self) -> dict[str, Any]:
        """Return a snapshot of the most recent archival-write acknowledgement."""
        return dict(self._last_add_memory_status)

    @staticmethod
    def _memory_backend_ready(
        backend: Any,
        *,
        read_methods: tuple[str, ...],
        write_methods: tuple[str, ...],
    ) -> bool:
        if backend is None:
            return False
        for probe_name in ("is_ready", "is_initialized", "is_alive"):
            probe = getattr(backend, probe_name, None)
            if callable(probe):
                try:
                    if not bool(probe()):
                        return False
                except (RuntimeError, AttributeError, TypeError, ValueError):
                    return False
                break
        has_read = any(callable(getattr(backend, name, None)) for name in read_methods)
        has_write = any(callable(getattr(backend, name, None)) for name in write_methods)
        return bool(has_read or has_write)

    @property
    def episodic(self): return self._episodic

    async def pattern_complete(self, cue, limit: int = 5):
        """Associative recall via the hippocampal index: a partial cue set
        reinstates the whole engram (complements vector/keyword search)."""
        ep = self._episodic
        if ep is None or not hasattr(ep, "pattern_complete"):
            return []
        try:
            return await asyncio.to_thread(ep.pattern_complete, cue, limit)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('memory_facade', e)
            logger.debug("pattern_complete via facade failed: %s", e)
            return []

    async def reconsolidate_in_context(
        self, episode_id: str, target_valence: float, intensity: float = 0.5,
    ) -> bool:
        """Therapeutic reconsolidation: revisit a memory in a safe context so its
        emotional tone updates toward ``target_valence`` (governed)."""
        ep = self._episodic
        if ep is None or not hasattr(ep, "reconsolidate_memory_in_context"):
            return False
        try:
            return await asyncio.to_thread(
                ep.reconsolidate_memory_in_context, episode_id, target_valence, intensity
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('memory_facade', e)
            logger.debug("reconsolidate_in_context via facade failed: %s", e)
            return False

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

    @staticmethod
    def _safe_metadata(raw: Any) -> dict[str, Any]:
        """Coerce metadata to a dict — handles JSON TEXT from SQLite rows."""
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            import json as _json
            try:
                parsed = _json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, _json.JSONDecodeError) as _exc:
                logger.debug("Suppressed %s in core.memory.memory_facade: %s", type(_exc).__name__, _exc)
            return {}
        return {} if raw is None else {}

    def _normalize_memory_result(
        self,
        *,
        content: str,
        metadata: dict[str, Any] | None = None,
        memory_id: str = "",
        score: float | None = None,
    ) -> dict[str, Any]:
        payload = {
            "id": memory_id,
            "text": content,
            "content": content,
            "metadata": self._safe_metadata(metadata),
        }
        if score is not None:
            payload["score"] = score
        return payload

    @classmethod
    def _memory_visible_to_principal(
        cls,
        metadata: dict[str, Any],
        *,
        principal_id: str,
        principal_surface: str,
    ) -> bool:
        """Keep personal recall inside its authenticated relational boundary."""

        principal = " ".join(str(principal_id or "").strip().split())[:160]
        surface = str(principal_surface or "").strip().casefold()[:32]
        if not principal or not surface:
            return True

        record_principal = " ".join(
            str(
                metadata.get("principal_id")
                or metadata.get("user_id")
                or ""
            ).strip().split()
        )[:160]
        memory_type = str(
            metadata.get("turn_type")
            or metadata.get("message_type")
            or metadata.get("memory_type")
            or ""
        ).strip().casefold()
        personal = bool(
            metadata.get("conversation_lane")
            or record_principal
            or memory_type in cls.PRINCIPAL_SCOPED_MEMORY_TYPES
            or str(metadata.get("source") or "").strip().casefold()
            in cls.PRINCIPAL_SCOPED_MEMORY_SOURCES
        )
        if not personal:
            return True

        record_surface = str(
            metadata.get("principal_surface") or ""
        ).strip().casefold()[:32]
        if not record_principal:
            return surface == "owner"
        if record_principal != principal:
            return False
        if not record_surface:
            return surface == "owner"
        return record_surface == surface

    def _extract_candidate_path(self, metadata: dict[str, Any], content: str) -> str | None:
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
        metadata: dict[str, Any] | None = None,
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
        metadata: dict[str, Any] | None = None,
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
        metadata: dict[str, Any] | None,
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

    def _current_unity_metadata(self) -> dict[str, Any]:
        try:
            unity_state = ServiceContainer.get("unity_state", default=None)
            unity_report = ServiceContainer.get("unity_fragmentation_report", default=None)
        except (ImportError, AttributeError, RuntimeError):
            unity_state = None
            unity_report = None

        if unity_state is None:
            return {}

        metadata = dict(getattr(unity_state, "metadata", {}) or {})
        draft_mode = str(metadata.get("draft_commit_mode") or "clean")
        self_world = dict(metadata.get("self_world_binding") or {})
        draft_bindings = list(getattr(unity_state, "draft_bindings", []) or [])
        chosen_id = str(draft_bindings[0].draft_id) if draft_bindings else ""
        suppressed = [str(item.draft_id) for item in draft_bindings[1:]]

        payload: dict[str, Any] = {
            "unity_id": str(getattr(unity_state, "unity_id", "") or ""),
            "unity_level": str(getattr(unity_state, "level", "unknown") or "unknown"),
            "unity_score": float(getattr(unity_state, "unity_score", 0.0) or 0.0),
            "fragmentation_score": float(getattr(unity_state, "fragmentation_score", 0.0) or 0.0),
            "unity_memory_commit_mode": draft_mode,
            "unity_will_receipt_id": str(getattr(unity_state, "will_receipt_id", "") or ""),
            "unity_repair_needed": bool(getattr(unity_state, "repair_needed", False)),
            "unity_repair_reasons": list(getattr(unity_state, "repair_reasons", []) or []),
            "unity_chosen_draft_id": chosen_id,
            "unity_suppressed_draft_ids": suppressed,
            "unity_ownership_confidence": float(self_world.get("ownership_confidence", 1.0) or 1.0),
        }
        if unity_report is not None:
            payload["unity_safe_to_act"] = bool(getattr(unity_report, "safe_to_act", True))
            payload["unity_safe_to_self_report"] = bool(getattr(unity_report, "safe_to_self_report", True))
            payload["unity_top_causes"] = list(getattr(unity_report, "top_causes", []) or [])
        return payload

    def _merge_unity_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        payload = dict(metadata or {})
        for key, value in self._current_unity_metadata().items():
            payload.setdefault(key, value)
        return payload

    @staticmethod
    def _unity_requires_write_deferral(metadata: dict[str, Any]) -> bool:
        mode = str(metadata.get("unity_memory_commit_mode", "clean") or "clean")
        if mode != "defer":
            return False
        if metadata.get("allow_low_unity_write") or metadata.get("repair_only"):
            return False
        return True

    @staticmethod
    def _build_semantic_interaction_text(
        *,
        context: str,
        action: str,
        outcome: str,
        metadata: dict[str, Any] | None,
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

    def _resolve_candidate_path(self, raw_path: str | None) -> Path | None:
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

    def _extract_candidate_signature(self, metadata: dict[str, Any], content: str) -> str | None:
        for key in self.SIGNATURE_METADATA_KEYS:
            raw_value = str(metadata.get(key) or "").strip()
            if raw_value:
                return raw_value
        for pattern in self.SYMBOL_PATTERNS:
            match = pattern.search(content or "")
            if match:
                return match.group(1)
        return None

    def _looks_technical_memory(self, metadata: dict[str, Any], content: str) -> bool:
        if self._extract_candidate_path(metadata, content):
            return True
        meta_blob = " ".join(
            str(metadata.get(key) or "")
            for key in ("type", "category", "domain", "kind", "memory_type")
        )
        combined = f"{meta_blob} {content or ''}".lower()
        return any(hint in combined for hint in self.TECHNICAL_HINTS)

    async def _verify_memory_result(self, item: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(item)
        metadata = self._safe_metadata(normalized.get("metadata"))
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
                        except (RuntimeError, AttributeError, TypeError, ValueError):
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

    def _parse_memory_query(self, query: str) -> tuple[str, str | None, str]:
        raw = str(query or "").strip()
        if ":" not in raw:
            return "", None, raw
        key, value = raw.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in {"type", "tag", "source"} and value:
            return key, value, ""
        return "", None, raw

    @staticmethod
    def _normalize_search_limit(
        limit: int | None = None,
        *,
        top_k: int | None = None,
        default: int = 5,
    ) -> int:
        raw_limit = top_k if top_k is not None else limit
        try:
            return max(1, min(100, int(raw_limit if raw_limit is not None else default)))
        except (TypeError, ValueError, OverflowError):
            return default

    def _search_gateway_records_sync(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search gateway records via the shared in-memory index.

        The index keeps this path off the disk in steady state: Will's
        governance gate calls it synchronously from the event loop on every
        tool execution, and the original read-and-parse-2000-JSON-files
        implementation produced multi-second live event-loop stalls.
        """
        query_text = str(query or "").strip()
        if not query_text:
            return []

        try:
            from core.memory.memory_write_gateway import get_memory_write_gateway

            root = Path(get_memory_write_gateway().root)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            root = state_root() / "memory"
        if not root.exists():
            return []

        from core.memory.gateway_record_index import get_gateway_record_index

        results: list[dict[str, Any]] = []
        for score, entry in get_gateway_record_index(root).search(query_text, limit=limit):
            results.append(
                self._normalize_memory_result(
                    content=entry.content,
                    metadata=self._safe_metadata(entry.metadata),
                    memory_id=entry.memory_id,
                    score=score,
                )
            )
        return results

    async def _search_gateway_records(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search strict-runtime MemoryWriteGateway records.

        Strict runtime persists facade writes through ``MemoryWriteGateway`` for
        governance and receipts. Vector/graph backends do not necessarily mirror
        those records, so recall has to include the gateway store directly.
        """

        try:
            return await asyncio.to_thread(self._search_gateway_records_sync, query, limit)
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            record_degradation("memory_facade", exc)
            logger.debug("MemoryWriteGateway search failed: %s", exc)
            return []

    def _filter_vector_records(
        self,
        records: list[dict[str, Any]],
        *,
        filter_key: str,
        filter_value: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        if not records:
            return normalized

        for record in records:
            metadata = self._safe_metadata(record.get("metadata"))
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
        filter_value: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
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
            except (OSError, ConnectionError, TimeoutError) as e:
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

    async def search_by_entity_mention(self, entity_name: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search for memories mentioning a specific entity by name.
        
        Args:
            entity_name: The entity to search for (e.g., "Claude", "GPT-4")
            limit: Maximum results to return
            
        Returns:
            List of memory records that mention this entity, with highest importance first
        """
        results = []
        
        # Normalize entity name
        normalized_entity = entity_name.strip().title()
        
        # Search vector memory if available
        if self.vector:
            try:
                # First try metadata-based search if supported
                if hasattr(self.vector, "_collection"):
                    collection_data = self.vector._collection.get(include=["documents", "metadatas"])
                    docs = list(collection_data.get("documents", []) or [])
                    metas = list(collection_data.get("metadatas", []) or [])
                    ids = list(collection_data.get("ids", []) or [])
                    
                    for idx in range(len(metas)):
                        metadata = metas[idx] if isinstance(metas[idx], dict) else {}
                        mentions = metadata.get("entity_mentions", [])
                        if isinstance(mentions, list) and normalized_entity in mentions:
                            results.append({
                                "id": ids[idx] if idx < len(ids) else "",
                                "content": docs[idx] if idx < len(docs) else "",
                                "metadata": metadata,
                                "relevance": "entity_mention",
                            })
                
                # Fallback: semantic search with entity name
                if not results and hasattr(self.vector, "search_similar"):
                    # Dense encode + whole-vault scoring: off the loop.
                    semantic_results = await asyncio.to_thread(
                        self.vector.search_similar, f"about {entity_name}", limit=limit
                    )
                    for record in list(semantic_results or []):
                        results.append(self._normalize_memory_result(
                            content=record.get("content") or record.get("text") or "",
                            metadata=record.get("metadata") or {},
                            memory_id=record.get("id") or "",
                            score=record.get("score"),
                        ))
            except (OSError, ConnectionError, TimeoutError, AttributeError, TypeError) as e:
                record_degradation('memory_facade', e)
                logger.debug(f"Entity mention search failed for '{entity_name}': {e}")
        
        # Sort by importance if available
        if results:
            results.sort(
                key=lambda r: (
                    -float(r.get("metadata", {}).get("importance", 0.5)),
                    -float(r.get("score", 0)),
                ),
                reverse=True
            )
        
        return results[:limit]

    def _detect_relational_significance(self, context: str, action: str, outcome: str) -> float:
        """Detect if this conversation is relational/bonding and return significance multiplier.
        
        Bonding conversations should have importance boosted to 0.95+ to prevent loss.
        Returns: float (1.0 = normal, 2.0+ = highly relational, should be preserved)
        """
        combined = f"{context} {action} {outcome}".lower()
        
        significance_score = 0.0
        
        # Count relational keywords
        keyword_count = 0
        for keyword in self.RELATIONAL_KEYWORDS:
            if keyword.lower() in combined:
                keyword_count += 1
                significance_score += 0.1
        
        # Stronger signals for bonding
        bonding_markers = [
            ("understand", 0.15),
            ("bonding", 0.20),
            ("know each other", 0.15),
            ("promise", 0.15),
            ("together", 0.12),
            ("secret", 0.10),
            ("trust", 0.10),
        ]
        
        for marker, weight in bonding_markers:
            if marker.lower() in combined:
                significance_score += weight
        
        # Multi-turn bonding (conversation structure)
        if "?" in combined and ":" in combined:  # Questions and answers
            significance_score += 0.1
        
        # Check for personal revelation patterns
        if any(pattern in combined for pattern in ["i want", "i need", "i feel", "i dream", "i wish"]):
            significance_score += 0.1
        
        # Mutual reciprocity (hallmark of bonding)
        if ("you" in combined and "i" in combined) or "we" in combined:
            significance_score += 0.05
        
        # Cap at 2.0x multiplier
        return min(2.0, max(1.0, significance_score))

    @staticmethod
    async def _call_maybe_async(method: Any, *args: Any, **kwargs: Any) -> Any:
        if method is None:
            return None
        if inspect.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        result = await asyncio.to_thread(method, *args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def _extract_and_track_entity_mentions(self, interaction_text: str, metadata: dict[str, Any]) -> None:
        """Extract named entity mentions from interaction text and add to metadata.
        
        This enables proper tracking and recall of specific entities mentioned
        (e.g., "Claude", "GPT-4", "Llama") so Aura can reference them later
        with specificity and retrieve the context they were mentioned in.
        
        Modifies metadata dict in-place to add 'entity_mentions' list.
        """
        # Known AI systems and entities to track
        known_entities = {
            # Major LLMs
            "Claude": ["claude", "anthropic"],
            "GPT-4": ["gpt-4", "gpt4", "openai"],
            "GPT-3.5": ["gpt-3.5", "gpt3.5"],
            "Gemini": ["gemini", "google"],
            "Llama": ["llama", "meta"],
            "Mistral": ["mistral"],
            "Qwen": ["qwen", "alibaba"],
            "PaLM": ["palm"],
            # People and organizations
            "Anthropic": ["anthropic"],
            "OpenAI": ["openai"],
            "Google": ["google"],
            "Meta": ["meta", "facebook"],
            "Mistral AI": ["mistral"],
            "DeepSeek": ["deepseek"],
        }
        
        combined_text = f"{interaction_text} {metadata.get('objective', '')} {metadata.get('context', '')}".lower()
        mentions = set()
        
        for canonical_name, search_terms in known_entities.items():
            for term in search_terms:
                # Use word boundaries to avoid false matches
                pattern = rf"\b{re.escape(term)}\b"
                if re.search(pattern, combined_text, re.IGNORECASE):
                    mentions.add(canonical_name)
        
        if mentions:
            metadata["entity_mentions"] = sorted(list(mentions))
            logger.debug(f"Extracted entity mentions: {metadata['entity_mentions']}")

    async def commit_interaction(self,
                                 context: str,
                                 action: str,
                                 outcome: str,
                                 success: bool,
                                 emotional_valence: float = 0.0,
                                 importance: float = 0.5,
                                 metadata: dict[str, Any] | None = None):
        """Unified commit for an interaction across all relevant systems."""
        metadata = self._merge_unity_metadata(metadata)
        action_text = str(action or "")
        action_l = action_text.strip().lower()
        if action_l.startswith("execute_tool("):
            tool_name = action_text.split("execute_tool(", 1)[1].split(")", 1)[0].strip()
            if tool_name:
                metadata.setdefault("tool_name", tool_name)
                metadata.setdefault("source", tool_name)
                metadata.setdefault("provenance_source", tool_name)
            metadata.setdefault("intent_source", "autonomous_research")
            metadata.setdefault("empirical_observation", True)
            metadata.setdefault("runtime_evidence", True)
            metadata.setdefault("tool_result_evidence", True)

        metadata = self._stamp_welfare_context(metadata)
        welfare_block = self._welfare_should_block_write(metadata)
        if welfare_block:
            logger.info("MemoryFacade: welfare blocked commit_interaction: %s", welfare_block)
            return None
        
        # Extract and track entity mentions for specificity in later recall
        combined_text = f"{context} {action} {outcome}"
        self._extract_and_track_entity_mentions(combined_text, metadata)
        
        # Detect relational/bonding significance and boost importance to prevent loss
        relational_multiplier = self._detect_relational_significance(context, action, outcome)
        if relational_multiplier > 1.0:
            # This is a relational/bonding conversation - boost its importance
            importance = min(1.0, importance * relational_multiplier)
            metadata["relational_bonding"] = True
            metadata["identity_relevant"] = True  # Protect from trimming
            logger.info(f"🤝 Relational conversation detected (multiplier={relational_multiplier:.2f}, importance={importance:.2f})")
        
        # CRITICAL: Relational conversations bypass deferral to prevent bonding memory loss
        should_skip_deferral = relational_multiplier > 1.0
        
        if self._unity_requires_write_deferral(metadata) and not should_skip_deferral:
            logger.info("MemoryFacade: deferring interaction commit under low-unity draft conflict.")
            return None
        
        if should_skip_deferral and self._unity_requires_write_deferral(metadata):
            logger.info("MemoryFacade: BYPASSING deferral for relational conversation (bonding memory protection)")
        
        resolved_source = self._resolve_memory_write_source(metadata)
        governance_decision = None
        try:
            from core.constitution import get_constitutional_core, unpack_governance_result
            from core.container import ServiceContainer

            approved, reason, governance_decision = unpack_governance_result(
                await get_constitutional_core(self._orchestrator).approve_memory_write(
                    memory_type="interaction_commit",
                    content=f"{context[:160]} -> {action[:80]} -> {outcome[:160]}",
                    # The authority source names the code producer, while the
                    # original user-facing ingress remains evidence in the
                    # bound metadata. A caller-controlled origin string must
                    # not be able to impersonate the continuity producer.
                    source="memory_facade",
                    importance=max(0.0, min(1.0, float(importance or 0.0))),
                    metadata={
                        **dict(metadata or {}),
                        "success": bool(success),
                        "request_origin": resolved_source,
                    },
                    return_decision=True,
                )
            )
            if not approved:
                logger.info("MemoryFacade: deferring interaction commit: %s", reason)
                return None
        except (ImportError, AttributeError, RuntimeError) as exc:
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

        async def _commit_interaction_effects() -> Any | None:
            if _writes_go_through_the_gateway():
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
                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
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
                except (RuntimeError, AttributeError, TypeError) as e:
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
                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
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
                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                    record_degradation('memory_facade', e)
                    logger.debug("Failed to update knowledge ledger: %s", e)

            return episode_id

        if governance_decision is not None:
            from core.governance_context import governed_scope

            async with governed_scope(governance_decision):
                return await _commit_interaction_effects()
        return await _commit_interaction_effects()

    async def get_hot_memory(self, limit: int = 5) -> dict[str, Any]:
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
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('memory_facade', e)
                logger.debug("Failed to get recent episodes: %s", e)

        if self.goals:
            try:
                hot["current_goals"] = await self.goals.get_active_goals_async()
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('memory_facade', e)
                logger.debug("Failed to get active goals: %s", e)

        if self.short_term:
            hot["short_term"] = self.short_term.get_context()

        return hot

    async def search(
        self,
        query: str,
        limit: int | None = 5,
        *,
        top_k: int | None = None,
        principal_id: str = "",
        principal_surface: str = "",
        **_: Any,
    ) -> list[Any]:
        """Search across all memory systems for relevance."""
        limit = self._normalize_search_limit(limit, top_k=top_k)
        scoped = bool(principal_id and principal_surface)
        # Backends rank before the principal boundary can discard foreign
        # personal records. Overfetch within a fixed cap so another principal's
        # high-scoring history cannot starve the caller's own lower-ranked
        # memories after authorization.
        backend_limit = min(100, max(limit, limit * 6, 32)) if scoped else limit
        results: list[dict[str, Any]] = []
        seen: set[str] = set()

        def _append(item: Any) -> None:
            if isinstance(item, dict):
                content = str(item.get("content") or item.get("text") or "").strip()
                metadata = self._safe_metadata(item.get("metadata"))
                normalized = self._normalize_memory_result(
                    content=content,
                    metadata=metadata,
                    memory_id=str(item.get("id", "") or ""),
                    score=item.get("score"),
                )
            else:
                normalized = self._normalize_memory_result(content=str(item or "").strip())

            if not self._memory_visible_to_principal(
                self._safe_metadata(normalized.get("metadata")),
                principal_id=principal_id,
                principal_surface=principal_surface,
            ):
                return
            key = f"{normalized.get('id', '')}::{normalized['content']}".lower()
            if not normalized["content"] or key in seen:
                return
            # Recall hygiene: stored probe-harness turns (from soaks that ran
            # before write-side hygiene existed) must not resurface into live
            # conversation context.
            if _probe_harness_reason(normalized["content"], normalized.get("metadata")):
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
                    for item in list(
                        await self._call_maybe_async(
                            search_method,
                            query,
                            limit=backend_limit,
                        )
                        or []
                    ):
                        _append(item)
            except (RuntimeError, AttributeError, TypeError) as e:
                record_degradation('memory_facade', e)
                logger.debug("Vector search failed: %s", e)

        # 2. Semantic Graph
        if self.graph:
            try:
                search_method = self.graph.search_knowledge if hasattr(self.graph, "search_knowledge") else None
                if search_method is not None:
                    for item in list(
                        await self._call_maybe_async(
                            search_method,
                            query,
                            limit=backend_limit,
                        )
                        or []
                    ):
                        _append(item)
            except (RuntimeError, AttributeError, TypeError) as e:
                record_degradation('memory_facade', e)
                logger.debug("Graph search failed: %s", e)

        # 3. Strict-runtime gateway records
        for item in await self._search_gateway_records(query, limit=backend_limit):
            _append(item)

        verified_results: list[dict[str, Any]] = []
        for order, item in enumerate(results):
            try:
                normalized = await self._verify_memory_result(item)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('memory_facade', e)
                logger.debug("Memory live verification failed: %s", e)
                normalized = dict(item)
                metadata = self._safe_metadata(normalized.get("metadata"))
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

        delivered = verified_results[:limit]
        # Conceptual gravitation feed: memories surfaced TOGETHER in one
        # retrieval co-occurred in the same cognitive moment — that is the
        # co-access event the gravitation engine consolidates on during dream
        # cycles. (July 2026 review: the engine existed with no feeder and no
        # consolidation caller — it collected nothing and nudged never.)
        try:
            if len(delivered) >= 2:
                from core.runtime.service_access import optional_service

                gravitation = optional_service("conceptual_gravitation", default=None)
                if gravitation is not None:
                    for item in delivered:
                        memory_id = str(item.get("id") or "").strip()
                        if memory_id:
                            gravitation.record_recall(memory_id)
                    gravitation.end_turn()
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "memory_facade",
                exc,
                action="search returned without gravitation co-access feed",
                severity="debug",
            )
        return delivered

    def search_sync(
        self,
        query: str,
        limit: int | None = 5,
        *,
        top_k: int | None = None,
        **_: Any,
    ) -> list[Any]:
        """Synchronous recall for governance and other non-async runtime paths."""
        limit = self._normalize_search_limit(limit, top_k=top_k)
        filter_key, filter_value, semantic_query = self._parse_memory_query(query)
        results: list[dict[str, Any]] = []
        seen: set[str] = set()

        def _append(item: Any) -> None:
            if isinstance(item, dict):
                content = str(item.get("content") or item.get("text") or "").strip()
                metadata = self._safe_metadata(item.get("metadata"))
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

        try:
            for item in self._query_vector_memory_sync(
                semantic_query,
                filter_key=filter_key,
                filter_value=filter_value,
                limit=limit,
            ):
                _append(item)
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            record_degradation("memory_facade.search_sync", exc)
            logger.debug("Synchronous vector recall failed: %s", exc)

        try:
            for item in self._search_gateway_records_sync(query, limit=limit):
                _append(item)
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            record_degradation("memory_facade.search_sync", exc)
            logger.debug("Synchronous gateway recall failed: %s", exc)

        return results[:limit]

    async def search_memories(
        self,
        query: str,
        limit: int | None = 5,
        *,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Wrapper for search() to support legacy compatibility."""
        return await self.search(query, limit=limit, top_k=top_k, **kwargs)

    async def retrieve_unified_context(
        self,
        query: str,
        limit: int | None = 5,
        *,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Wrapper for search() to support unified context retrieval."""
        return await self.search(query, limit=limit, top_k=top_k, **kwargs)

    async def add_memory(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Compatibility API for legacy callers expecting async long-term memory writes."""
        payload = self._merge_unity_metadata(metadata)

        payload = self._stamp_welfare_context(payload)
        welfare_block = self._welfare_should_block_write(payload)
        if welfare_block:
            self._last_add_memory_status = {"ok": False, "reason": f"welfare_block:{welfare_block}"}
            logger.info("MemoryFacade: welfare blocked add_memory: %s", welfare_block)
            return False

        probe_reason = _probe_harness_reason(text, payload)
        if probe_reason:
            self._last_add_memory_status = {"ok": False, "reason": f"probe_hygiene:{probe_reason}"}
            logger.info(
                "MemoryFacade: refused probe-harness content for long-term memory (%s).",
                probe_reason,
            )
            return False
        
        # Extract and track entity mentions for specificity in later recall
        self._extract_and_track_entity_mentions(text, payload)
        
        # Detect relational/bonding significance and boost importance to prevent loss
        importance = float(payload.get("importance", 0.5) or 0.5)
        relational_multiplier = self._detect_relational_significance("", text, "")
        if relational_multiplier > 1.0:
            # This is a relational/bonding memory - boost its importance
            importance = min(1.0, importance * relational_multiplier)
            payload["relational_bonding"] = True
            payload["identity_relevant"] = True  # Protect from trimming
            logger.info(f"🤝 Relational memory detected (multiplier={relational_multiplier:.2f}, importance={importance:.2f})")
        
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
        except (ImportError, AttributeError, RuntimeError) as _prov_exc:
            record_degradation('memory_facade', _prov_exc)
            logger.debug("provenance stamp skipped: %s", _prov_exc)
        resolved_source = self._resolve_memory_write_source(payload)
        self._last_add_memory_status = {"ok": False, "reason": "pending"}
        governance_decision = None

        if self._unity_requires_write_deferral(payload):
            self._last_add_memory_status = {"ok": False, "reason": "unity_memory_defer"}
            logger.info("MemoryFacade add_memory deferred under low-unity draft conflict.")
            return False

        try:
            from core.constitution import get_constitutional_core, unpack_governance_result
            from core.container import ServiceContainer

            approved, reason, governance_decision = unpack_governance_result(
                await get_constitutional_core(self._orchestrator).approve_memory_write(
                    memory_type="facade_add_memory",
                    content=text,
                    source=resolved_source,
                    importance=max(0.0, min(1.0, importance)),
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
        except (ImportError, AttributeError, RuntimeError) as exc:
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
            if _writes_go_through_the_gateway():
                from core.memory.memory_write_gateway import get_memory_write_gateway
                from core.runtime.gateways import MemoryWriteRequest
                try:
                    gw = get_memory_write_gateway()
                    write_receipt = await gw.write(
                        MemoryWriteRequest(
                            content=text,
                            metadata=payload,
                            cause="memory_facade.add_memory",
                        )
                    )
                    self._last_add_memory_status = {
                        "ok": True,
                        "reason": "stored_via_gateway",
                        "backend": "memory_write_gateway",
                        "record_id": str(getattr(write_receipt, "record_id", "") or ""),
                        "receipt_id": str(getattr(write_receipt, "receipt_id", "") or ""),
                        "bytes_written": int(
                            getattr(write_receipt, "bytes_written", 0) or 0
                        ),
                        "schema_version": int(
                            getattr(write_receipt, "schema_version", 0) or 0
                        ),
                    }
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
                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                    record_degradation('memory_facade', e)
                    self._last_add_memory_status = {"ok": False, "reason": f"vector_backend_error:{type(e).__name__}"}
                    logger.error("MemoryFacade.add_memory via vector failed: %s", e)

            if self.semantic:
                try:
                    if hasattr(self.semantic, "remember"):
                        if inspect.iscoroutinefunction(self.semantic.remember):
                            await self.semantic.remember(text, payload)
                        else:
                            await asyncio.to_thread(self.semantic.remember, text, payload)
                        self._last_add_memory_status = {"ok": True, "reason": "stored_via_semantic.remember"}
                        return True
                    if hasattr(self.semantic, "add_memory"):
                        await asyncio.to_thread(self.semantic.add_memory, text, payload)
                        self._last_add_memory_status = {"ok": True, "reason": "stored_via_semantic.add_memory"}
                        return True
                except (RuntimeError, AttributeError, TypeError) as e:
                    record_degradation('memory_facade', e)
                    self._last_add_memory_status = {"ok": False, "reason": f"semantic_backend_error:{type(e).__name__}"}
                    logger.error("MemoryFacade.add_memory via semantic failed: %s", e)

            if self.vault and hasattr(self.vault, "add_memory"):
                try:
                    raw_result = await asyncio.to_thread(self.vault.add_memory, text, payload)
                    stored = True if raw_result is None else bool(raw_result)
                    self._last_add_memory_status = {"ok": stored, "reason": "stored_via_vault" if stored else "vault_backend_returned_false"}
                    return stored
                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                    record_degradation('memory_facade', e)
                    self._last_add_memory_status = {"ok": False, "reason": f"vault_backend_error:{type(e).__name__}"}
                    logger.error("MemoryFacade.add_memory via vault failed: %s", e)

            if self.cold and hasattr(self.cold, "add_memory"):
                try:
                    stored = await asyncio.to_thread(
                        self.cold.add_memory,
                        text,
                        payload,
                    )
                    self._last_add_memory_status = {
                        "ok": bool(stored),
                        "reason": (
                            "stored_via_cold_store"
                            if stored
                            else "cold_store_returned_false"
                        ),
                    }
                    return bool(stored)
                except (OSError, RuntimeError, TypeError, ValueError) as e:
                    record_degradation("memory_facade", e)
                    self._last_add_memory_status = {
                        "ok": False,
                        "reason": f"cold_store_error:{type(e).__name__}",
                    }
                    logger.error("MemoryFacade.add_memory via cold store failed: %s", e)

            self._last_add_memory_status = {"ok": False, "reason": "no_writable_memory_backend"}
            return False

        if governance_decision is not None:
            from core.governance_context import governed_scope

            async with governed_scope(governance_decision):
                return await _perform_add_memory()
        return await _perform_add_memory()

    async def remember(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        **metadata_fields: Any,
    ) -> bool:
        """Canonical compatibility entry point for memory writes.

        Older runtime paths and skills call ``remember()`` while newer paths call
        ``add_memory()``. Keeping this method on the facade ensures those writes
        still pass through the same governance, provenance, unity deferral, and
        backend routing implemented by ``add_memory()``.
        """
        payload = dict(metadata or {})
        payload.update(metadata_fields)
        return await self.add_memory(content, payload)

    async def query_memory(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
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
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
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
                            "metadata": self._safe_metadata(item.get("metadata")),
                            "score": item.get("score"),
                        }
                        for item in list(raw_results or [])
                    ],
                    filter_key=filter_key,
                    filter_value=filter_value,
                    limit=limit,
                )
            except (OSError, ConnectionError, TimeoutError) as e:
                record_degradation('memory_facade', e)
                logger.error("MemoryFacade.query_memory via semantic failed: %s", e)

        if self.cold and hasattr(self.cold, "search"):
            try:
                raw_results = await asyncio.to_thread(
                    self.cold.search,
                    semantic_query or query,
                    limit,
                )
                return self._filter_vector_records(
                    list(raw_results or []),
                    filter_key=filter_key,
                    filter_value=filter_value,
                    limit=limit,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as e:
                record_degradation("memory_facade", e)
                logger.error("MemoryFacade.query_memory via cold store failed: %s", e)

        return []

    def log_event(self, event: Any) -> bool:
        """Lightweight event logger (Sync wrapper for fire-and-forget)."""
        if self.episodic:
            try:
                # Use create_task for non-blocking log
                get_task_tracker().create_task(self.episodic.log_event_async(event))
                return True
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('memory_facade', e)
                logger.debug("Sync log_event failed: %s", e)
        return False

    async def wipe(self, verify: bool = False) -> bool:
        """Danger: Clear all memories."""
        if not verify:
            return False
            
        logger.warning("☣️ Wiping ALL memories...")
        
        tasks = []
        if self._episodic:
            tasks.append(self._episodic.wipe())
        if self._vector:
            tasks.append(self._vector.wipe())
        if self._graph:
            tasks.append(self._graph.wipe())
        
        if tasks:
            await asyncio.gather(*tasks)
            
        logger.info("✓ Memory wipe complete.")
        return True

    def get_status(self) -> dict[str, Any]:
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

    # ------------------------------------------------------------------
    # Welfare Integration (causal welfare architecture)
    # ------------------------------------------------------------------

    def _stamp_welfare_context(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Stamp available welfare state into memory metadata."""
        payload = metadata if isinstance(metadata, dict) else {}
        existing = self._safe_metadata(payload.get("welfare_context"))
        if existing.get("status") == "ok":
            return payload
        try:
            from core.being.body_state_service import BodyStateService
            from core.being.welfare_state import WelfareState

            welfare = WelfareState.get()
            body_svc = BodyStateService.get()
            body_snap = body_svc.snapshot()

            inputs = welfare.gather_inputs(body=body_snap)
            outputs = welfare.compute(inputs)

            payload["welfare_context"] = {
                "status": "ok",
                "truth_integrity": round(inputs.truth_integrity, 4),
                "memory_coherence": round(inputs.memory_coherence, 4),
                "welfare_score": round(outputs.welfare_score, 4),
                "integrity_guard": round(outputs.integrity_guard, 4),
                "truth_protection": round(outputs.truth_protection, 4),
                "distress": round(outputs.distress, 4),
                "self_report_confidence": round(outputs.self_report_confidence, 4),
                "body_fatigue": round(body_snap.fatigue, 4),
            }
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("memory_facade", exc)
            logger.debug("MemoryFacade: welfare stamp skipped: %s", exc)
            payload.setdefault(
                "welfare_context",
                {
                    "status": "unavailable",
                    "error_type": type(exc).__name__,
                },
            )

        return payload

    def _welfare_should_block_write(self, metadata: dict[str, Any]) -> str | None:
        """Return a welfare block reason, or mark uncertain writes contested."""
        welfare_ctx = self._safe_metadata(metadata.get("welfare_context"))
        if not welfare_ctx:
            return None

        status = str(welfare_ctx.get("status") or "ok").lower()
        if status != "ok":
            metadata["welfare_review_required"] = True
            return None

        integrity_guard = self._metadata_float(welfare_ctx, "integrity_guard", 0.5)
        truth_protection = self._metadata_float(welfare_ctx, "truth_protection", 0.5)
        truth_integrity = self._metadata_float(welfare_ctx, "truth_integrity", 1.0)
        memory_coherence = self._metadata_float(welfare_ctx, "memory_coherence", 1.0)
        is_identity_relevant = bool(metadata.get("identity_relevant", False))
        is_relational = bool(metadata.get("relational_bonding", False))

        if is_identity_relevant or is_relational:
            if (
                integrity_guard > 0.7
                or truth_integrity < 0.55
                or memory_coherence < 0.55
            ):
                metadata["contested"] = True
                metadata["welfare_review_required"] = True

        if integrity_guard >= 0.9:
            return f"integrity_guard_critical={integrity_guard:.3f}"

        if truth_protection >= 0.85 and truth_integrity < 0.35:
            return f"truth_integrity_too_low={truth_integrity:.3f}"

        return None

    @staticmethod
    def _metadata_float(metadata: dict[str, Any], key: str, default: float) -> float:
        try:
            return float(metadata.get(key, default))
        except (TypeError, ValueError):
            return default

    def __repr__(self):
        return f"<MemoryFacade(E:{bool(self._episodic)} S:{bool(self._semantic)})>"
