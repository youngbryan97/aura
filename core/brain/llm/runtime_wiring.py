from __future__ import annotations

import asyncio
import inspect
import logging
import os
from typing import Any

from core.phases.response_contract import ResponseContract, build_response_contract
from core.runtime import service_access
from core.runtime.errors import FallbackClassification, Severity, record_degradation

logger = logging.getLogger(__name__)

_SOFT_RUNTIME_FAILURES = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

_USER_FACING_ORIGINS = frozenset(
    {
        "user",
        "voice",
        "admin",
        "api",
        "gui",
        "ws",
        "websocket",
        "direct",
        "external",
        "audit",
        "simulate",
    }
)


def _record_runtime_wiring_degradation(
    error: BaseException,
    *,
    stage: str,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {"stage": stage, "repair_requested": True}
    if extra:
        payload.update(extra)
    record_degradation(
        "runtime_wiring",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        extra=payload,
    )


def _origin_tokens(origin: str | None) -> set[str]:
    normalized = str(origin or "").strip().lower().replace("-", "_")
    return {token for token in normalized.split("_") if token}


def is_user_facing_origin(origin: str | None) -> bool:
    return bool(_origin_tokens(origin) & _USER_FACING_ORIGINS)


def _objective_from_messages(messages: list[dict[str, Any]] | None) -> str:
    if not messages:
        return ""
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "") or "").strip().lower()
        if role in {"user", "human"}:
            return str(msg.get("content", "") or "").strip()
    return ""


def _coerce_prompt_from_messages(messages: list[dict[str, Any]] | None) -> tuple[str, str | None]:
    if not messages:
        return "", None

    system_parts: list[str] = []
    convo_parts: list[str] = []

    for msg in messages:
        if not isinstance(msg, dict):
            convo_parts.append(str(msg))
            continue

        role = str(msg.get("role", "") or "").strip().lower()
        content = str(msg.get("content", "") or "").strip()
        if not content:
            continue

        if role == "system":
            system_parts.append(content)
        elif role in {"user", "human"}:
            convo_parts.append(f"User: {content}")
        elif role in {"assistant", "aura"}:
            convo_parts.append(f"Aura: {content}")
        else:
            convo_parts.append(f"[{role or 'message'}]: {content}")

    prompt = "\n".join(convo_parts).strip()
    system_prompt = "\n\n".join(system_parts).strip() or None
    return prompt, system_prompt


def _merge_system_prompt(messages: list[dict[str, Any]], extra: str) -> list[dict[str, Any]]:
    if not extra:
        return messages

    merged = [dict(m) if isinstance(m, dict) else m for m in messages]
    if merged and isinstance(merged[0], dict) and merged[0].get("role") == "system":
        base = str(merged[0].get("content", "") or "").strip()
        normalized_extra = str(extra or "").strip()
        if base == normalized_extra or base.startswith(f"{normalized_extra}\n\n"):
            return merged
        merged[0]["content"] = f"{extra}\n\n{base}" if base else extra
        return merged

    return [{"role": "system", "content": extra}, *merged]


async def resolve_runtime_state(
    explicit_state: Any = None,
    *,
    origin: str | None,
    is_background: bool,
) -> Any:
    if explicit_state is not None and hasattr(explicit_state, "cognition"):
        return explicit_state
    if is_background or not is_user_facing_origin(origin):
        return None

    try:
        repo = service_access.resolve_state_repository(default=None)
        if repo and hasattr(repo, "get_current"):
            return await repo.get_current()
    except _SOFT_RUNTIME_FAILURES as exc:
        _record_runtime_wiring_degradation(
            exc,
            stage="state_repository_resolution",
            action="continued with explicit payload inputs because live state hydration was unavailable",
            severity="degraded",
            extra={"origin": str(origin or "system"), "is_background": is_background},
        )
        return None
    return None


def _normalize_memory_snippet(item: Any) -> str:
    if isinstance(item, dict):
        content = str(item.get("content") or item.get("text") or "").strip()
        raw_meta = item.get("metadata")
        if isinstance(raw_meta, str):
            import json as _json

            try:
                raw_meta = _json.loads(raw_meta)
            except (ValueError, _json.JSONDecodeError):
                raw_meta = {}
        metadata = raw_meta if isinstance(raw_meta, dict) else {}
        memory_type = str(metadata.get("type", "") or "").strip().lower()
        prefix = (
            f"[{memory_type}] "
            if memory_type in {"fact", "preference", "recent_episode", "shared_ground"}
            else ""
        )
        return f"{prefix}{content}".strip()
    return str(item or "").strip()


async def _call_memory_method(method: Any, *args: Any, **kwargs: Any) -> Any:
    if method is None:
        return None
    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    result = await asyncio.to_thread(method, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def _hydrate_runtime_memory(payload_state: Any, objective: str) -> None:
    if payload_state is None or not objective:
        return

    snippets: list[str] = []
    seen: set[str] = set()

    def _push(item: Any) -> None:
        snippet = _normalize_memory_snippet(item)
        if not snippet:
            return
        key = snippet.lower()
        if key in seen:
            return
        seen.add(key)
        snippets.append(snippet)

    for item in list(getattr(payload_state.cognition, "long_term_memory", []) or []):
        _push(item)

    try:
        memory = service_access.resolve_memory_facade(default=None)
        if memory is not None:
            search_method = getattr(memory, "search", None)
            if search_method is not None:
                for item in list(
                    await _call_memory_method(search_method, objective, limit=5) or []
                ):
                    _push(item)

            hot_method = getattr(memory, "get_hot_memory", None)
            if hot_method is not None:
                hot = await _call_memory_method(hot_method, limit=3)
                if isinstance(hot, dict):
                    for episode in list(hot.get("recent_episodes", []) or []):
                        _push({"content": episode, "metadata": {"type": "recent_episode"}})

        if not snippets:
            graph = service_access.optional_service("knowledge_graph", default=None)
            search_knowledge = (
                getattr(graph, "search_knowledge", None) if graph is not None else None
            )
            if search_knowledge is not None:
                for item in list(
                    await _call_memory_method(search_knowledge, objective, limit=3) or []
                ):
                    _push(item)
    except _SOFT_RUNTIME_FAILURES as exc:
        _record_runtime_wiring_degradation(
            exc,
            stage="runtime_memory_hydration",
            action="continued payload assembly with existing state memory after retrieval hydration failed",
            severity="degraded",
            extra={"objective_preview": objective[:160]},
        )
        return

    if snippets:
        payload_state.cognition.long_term_memory = snippets[:8]


async def prepare_runtime_payload(
    *,
    prompt: str | None,
    system_prompt: str | None,
    messages: list[dict[str, Any]] | None,
    state: Any,
    origin: str | None,
    is_background: bool,
) -> tuple[str, str | None, list[dict[str, Any]] | None, ResponseContract | None, Any]:
    objective = str(prompt or _objective_from_messages(messages) or "").strip()
    runtime_state = await resolve_runtime_state(state, origin=origin, is_background=is_background)
    contract: ResponseContract | None = None
    prepared_messages = messages

    if runtime_state is not None and objective:
        payload_state = runtime_state
        try:
            if hasattr(runtime_state, "derive"):
                payload_state = runtime_state.derive("runtime_llm_payload", origin="runtime_wiring")
        except _SOFT_RUNTIME_FAILURES as exc:
            _record_runtime_wiring_degradation(
                exc,
                stage="payload_state_derivation",
                action="using original runtime state because derived LLM payload clone failed",
                severity="warning",
                extra={"origin": str(origin or "system")},
            )
            payload_state = runtime_state

        try:
            payload_state.cognition.current_objective = objective
            payload_state.cognition.current_origin = str(origin or "system")
        except _SOFT_RUNTIME_FAILURES as _exc:
            _record_runtime_wiring_degradation(
                _exc,
                stage="payload_state_stamping",
                action="continued with unstamped runtime state; response contract will be built from explicit objective",
                severity="degraded",
                extra={"origin": str(origin or "system"), "objective_preview": objective[:160]},
            )
            logger.debug("Runtime payload state stamp skipped: %s", _exc)

        if not is_background:
            try:
                from core.voice.substrate_voice_engine import get_substrate_voice_engine

                get_substrate_voice_engine().compile_profile(
                    state=payload_state,
                    user_message=str(objective or "")[:500],
                    origin=str(origin or "system"),
                )
            except _SOFT_RUNTIME_FAILURES as exc:
                _record_runtime_wiring_degradation(
                    exc,
                    stage="substrate_voice_profile",
                    action="continued without precompiled substrate voice profile; downstream sampler/voice gates remain authoritative",
                    severity="warning",
                    extra={"origin": str(origin or "system"), "objective_preview": objective[:160]},
                )
                logger.debug("Substrate profile precompile skipped: %s", exc)

        if not is_background:
            try:
                await _hydrate_runtime_memory(payload_state, objective)
            except _SOFT_RUNTIME_FAILURES as _exc:
                _record_runtime_wiring_degradation(
                    _exc,
                    stage="payload_memory_hydration",
                    action="continued payload assembly with pre-existing memory evidence only",
                    severity="degraded",
                    extra={"origin": str(origin or "system"), "objective_preview": objective[:160]},
                )
                logger.debug("Runtime memory hydration skipped: %s", _exc)

        try:
            contract = build_response_contract(
                payload_state,
                objective,
                is_user_facing=not is_background and is_user_facing_origin(origin),
            )
        except _SOFT_RUNTIME_FAILURES as exc:
            _record_runtime_wiring_degradation(
                exc,
                stage="response_contract",
                action="continued without a response contract after contract construction failed",
                severity="critical",
                extra={"origin": str(origin or "system"), "objective_preview": objective[:160]},
            )
            contract = None

        if prepared_messages is None and not is_background:
            try:
                from core.brain.llm.context_assembler import ContextAssembler

                prepared_messages = ContextAssembler.build_messages(payload_state, objective)
            except _SOFT_RUNTIME_FAILURES as exc:
                _record_runtime_wiring_degradation(
                    exc,
                    stage="context_assembly",
                    action="using raw prompt/messages because context assembler failed",
                    severity="degraded",
                    extra={"origin": str(origin or "system"), "objective_preview": objective[:160]},
                )
                prepared_messages = None

    if prepared_messages is not None:
        if system_prompt:
            prepared_messages = _merge_system_prompt(prepared_messages, system_prompt)
        if contract and contract.reason != "ordinary_dialogue":
            prepared_messages = _merge_system_prompt(
                prepared_messages, contract.to_prompt_block().strip()
            )
        prompt, inferred_system = _coerce_prompt_from_messages(prepared_messages)
        system_prompt = inferred_system or system_prompt
    elif contract and contract.reason != "ordinary_dialogue":
        block = contract.to_prompt_block().strip()
        system_prompt = f"{system_prompt}\n\n{block}".strip() if system_prompt else block

    return str(prompt or ""), system_prompt, prepared_messages, contract, runtime_state


def derive_substrate_generation_overrides(
    *,
    runtime_state: Any,
    objective: str,
    origin: str | None,
    is_background: bool,
) -> dict[str, Any]:
    """Compile substrate-driven sampler overrides for foreground generation."""
    if runtime_state is None or is_background or not objective:
        return {}

    try:
        from core.voice.substrate_voice_engine import get_substrate_voice_engine

        sve = get_substrate_voice_engine()
        sve.compile_profile(
            state=runtime_state,
            user_message=str(objective or "")[:500],
            origin=str(origin or "system"),
        )
        overrides = dict(sve.get_generation_params() or {})
        if overrides:
            overrides["substrate_generation_source"] = str(
                getattr(sve.get_current_profile(), "compilation_source", "") or "substrate_voice"
            )
        return overrides
    except _SOFT_RUNTIME_FAILURES as exc:
        _record_runtime_wiring_degradation(
            exc,
            stage="substrate_generation_overrides",
            action="continued with caller/default generation parameters because substrate override compilation failed",
            severity="warning",
            extra={"origin": str(origin or "system"), "objective_preview": objective[:160]},
        )
        logger.debug("Substrate generation override skipped: %s", exc)
        return {}


def build_agentic_tool_map(
    required_skill: str | None = None,
    *,
    objective: str | None = None,
    max_tools: int = 8,
) -> dict[str, Any] | None:
    try:
        from core.container import ServiceContainer

        cap = ServiceContainer.get("capability_engine", default=None)
        if not cap or not hasattr(cap, "get_tool_definitions"):
            return None

        if hasattr(cap, "select_tool_definitions"):
            tool_defs = (
                cap.select_tool_definitions(
                    objective=str(objective or ""),
                    required_skill=required_skill,
                    max_tools=max_tools,
                )
                or []
            )
        else:
            tool_defs = cap.get_tool_definitions() or []
        tools: dict[str, Any] = {}
        for entry in tool_defs:
            fn = entry.get("function", {}) if isinstance(entry, dict) else {}
            name = fn.get("name")
            if not name:
                continue
            if required_skill and name != required_skill:
                if str(name) != str(required_skill):
                    continue
            tools[name] = fn
        return tools or None
    except _SOFT_RUNTIME_FAILURES as exc:
        _record_runtime_wiring_degradation(
            exc,
            stage="agentic_tool_map",
            action="returned no agentic tool map after capability registry lookup failed",
            severity="degraded",
            extra={
                "required_skill": str(required_skill or ""),
                "objective_preview": str(objective or "")[:160],
                "max_tools": max_tools,
            },
        )
        return None


def should_force_tool_handoff(contract: ResponseContract | None, *, is_background: bool) -> bool:
    if os.environ.get("AURA_EMBODIED_CHALLENGE"):
        return False
    return bool(
        contract
        and contract.requires_search
        and not contract.tool_evidence_available
        and not is_background
    )
