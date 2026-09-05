from __future__ import annotations

from collections.abc import Sequence

import asyncio
import inspect
import logging
import os
import re
from typing import Any

from core.dialogue.referents import UNATTRIBUTED, attribute, speaker_of
from core.memory.experience_provenance import provenance_label
from core.phases.response_contract import (
    ResponseContract,
    build_response_contract,
    looks_like_capability_inventory_request,
)
from core.runtime import service_access
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from dataclasses import dataclass

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

_MEMORY_HYDRATION_REQUEST_RE = re.compile(
    r"\b(?:"
    r"remember|recall|memory|memories|memor(?:y|ies)|earlier|previous|last time|"
    r"what did (?:i|you|we)|what have (?:i|you|we)|across sessions|"
    r"relationship|between us|our dynamic|dynamic changed|changed between us|evolved between us|"
    r"ground(?:ed|ing)?|evidence|receipt|source|cite|search|look up|web|browser|"
    r"open|create|write|save|export|file|folder|document|note|tool|tools"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ContractUnavailable:
    """A response contract that could not be built, in the slot where one goes.

    `contract = None` meant two things: "this turn needs no contract" (a
    background turn, the common case) and "contract construction FAILED and
    the turn is being generated without the rules that make the reply hers".
    The second was recorded at critical severity and then handed to the
    caller as the same `None` as the first, so nothing downstream could tell
    a deliberate absence from a lost one.

    Deliberately FALSY. Every existing `if contract:` and
    `if contract and contract.reason != ...` branch treats it exactly as it
    treated `None`, so generation behaviour is unchanged — this adds a
    distinction without moving the code that reads it. A caller that wants
    to know asks `contract is not None` or reads `.error`.

    Returned rather than raised: a user-facing turn with no contract is
    worse than one with, and better than no reply at all.
    """

    reason: str
    error: str = ""

    def __bool__(self) -> bool:
        return False


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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _memory_hydration_timeout_s() -> float:
    return max(0.05, _env_float("AURA_RUNTIME_MEMORY_HYDRATION_TIMEOUT_S", 1.25))


def _should_hydrate_runtime_memory(objective: str, origin: str | None) -> bool:
    text = str(objective or "").strip()
    if not text:
        return False
    if not is_user_facing_origin(origin):
        return True
    return bool(_MEMORY_HYDRATION_REQUEST_RE.search(text))


def is_user_facing_origin(origin: str | None) -> bool:
    """Delegate to the canonical foreground-origin classifier.

    This module carried a third, naive copy of the rule: it split the origin
    on separators and returned True if ANY token intersected a public set
    containing "api", "admin", "external", "audit", "simulate" and "test".
    A composite label could therefore self-declare user-facing authority —
    "test_generator" and "audit"/"simulate" were classified as real user
    traffic, which drives live-state resolution, response-contract
    construction, and memory hydration. The same naive rule also MISSED
    "native-shell", a genuinely user-facing desktop origin, because no token
    matched.

    core.goals.objective_lifecycle is the single source of truth: exact
    membership plus prefix anchoring ("desktop_", "voice_", "api_", ...), so
    "desktop_task" and "chat_api" still qualify while "test_generator" and
    "background_ui" do not.
    """
    from core.goals.objective_lifecycle import is_foreground_objective_origin

    return bool(is_foreground_objective_origin(origin))


# Turn boundaries in the flattened transcript are literal text, so these
# markers appearing INSIDE message content can forge a boundary and reassign
# authorship of everything that follows.
_ROLE_MARKER_RE = re.compile(
    r"(?im)^[ \t]*(?:user|human|aura|assistant|system)[ \t]*:",
)
_CHAT_CONTROL_TOKEN_RE = re.compile(
    r"(?i)<\|(?:im_start|im_end|endoftext|eot_id)\|>",
)


def _neutralize_role_markers(content: str) -> str:
    """Defuse embedded role labels and chat-control tokens in message text."""
    cleaned = _CHAT_CONTROL_TOKEN_RE.sub("", str(content or ""))
    # Zero-width word joiner after the label keeps the text readable to a
    # human while removing its line-anchored "new turn" shape.
    return _ROLE_MARKER_RE.sub(lambda m: m.group(0).replace(":", "⁠:"), cleaned)


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
            # A non-mapping item has no role to trust. Stringifying it into
            # the transcript let arbitrary objects contribute unattributed
            # text; it is skipped rather than silently promoted.
            _record_runtime_wiring_degradation(
                TypeError(f"non_mapping_message:{type(msg).__name__}"),
                stage="message_coercion",
                action="dropped a non-mapping message instead of stringifying it into the prompt",
            )
            continue

        role = str(msg.get("role", "") or "").strip().lower()
        content = str(msg.get("content", "") or "").strip()
        if not content:
            continue

        # Flattening roles into "User:"/"Aura:" lines means the ONLY thing
        # separating turns is literal text, so content that itself contains
        # those labels or chat-control tokens can forge a turn boundary and
        # reassign authorship of everything after it. Neutralize the markers
        # inside content before it becomes a labeled line.
        content = _neutralize_role_markers(content)

        if role == "system":
            system_parts.append(content)
        elif role in {"user", "human"}:
            convo_parts.append(f"User: {content}")
        elif role in {"assistant", "aura"}:
            convo_parts.append(f"Aura: {content}")
        else:
            # Unknown roles are rendered without an authority-bearing label.
            convo_parts.append(f"[unverified {role or 'message'}]: {content}")

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
        source = str(
            metadata.get("source")
            or metadata.get("origin")
            or metadata.get("provenance")
            or ""
        ).strip()
        return _render_recalled_snippet(
            content,
            memory_type=memory_type,
            source=source,
            speaker=speaker_of(metadata),
            metadata_for_provenance=metadata,
        )
    return _render_recalled_snippet(
        str(item or ""),
        memory_type="",
        source="",
        speaker=UNATTRIBUTED,
        metadata_for_provenance=None,
    )


# Memory content is RECALLED DATA, not instruction. Much of what reaches the
# store originates outside Aura — pages she read, documents and pasted text —
# so anything embedded in it that looks like a turn boundary or a directive
# must not be able to act as one when it is replayed into the prompt.
_RECALL_OPEN = "<recalled"
_RECALL_CLOSE = "</recalled>"
_MAX_SNIPPET_CHARS = 1200


def _render_recalled_snippet(
    content: Any,
    *,
    memory_type: str,
    source: str,
    speaker: str = UNATTRIBUTED,
    metadata_for_provenance: Any = None,
) -> str:
    """Render one memory as quoted, attributed, instruction-inert text.

    CP126 1983010a. Snippets used to be the raw stored string with at most a
    cosmetic "[type] " prefix: no source identity, no trust marker, no
    quoting boundary and no instruction neutralization before
    ContextAssembler consumed them. Text recalled from a web page could
    therefore arrive in the prompt indistinguishable from Aura's own
    reasoning, carrying whatever directives it liked.

    Three things travel with every snippet now:

    * a **boundary**, so the model can see where recalled data starts and
      ends rather than inferring it from a prefix;
    * **provenance** — what kind of memory this is and where it came from —
      so a claim can be attributed instead of absorbed; and
    * **neutralization** of role markers and chat-control tokens, the same
      defusing already applied to message content, so embedded turn
      boundaries cannot forge conversational authority.
    """
    text = _neutralize_role_markers(str(content or "").strip())
    if not text:
        return ""
    # A snippet cannot smuggle a fake closing tag to escape its own boundary.
    text = text.replace(_RECALL_CLOSE, "").replace(_RECALL_OPEN, "")
    if len(text) > _MAX_SNIPPET_CHARS:
        text = text[:_MAX_SNIPPET_CHARS].rstrip() + " …"
    attributes = []
    # WHO SAID IT comes first, because it is the attribute that decides what
    # the sentence means. A recalled "I was trying to get you to write about
    # yourself" is a different claim from Bryan's mouth than from hers, and
    # for one live evening it travelled bare and she read it as her own. An
    # unattributed snippet says so out loud rather than defaulting to her
    # voice — but only when the text actually turns on a pronoun; "the folder
    # is on the Desktop" means the same thing from anyone.
    label = attribute(text, speaker)
    if label:
        attributes.append(f'speaker="{_sanitize_attribute(label)}"')
    # ...and WHETHER IT HAPPENED. A journal entry, a narrative arc and a dream
    # all arrived here with no type attribute at all, which made them
    # indistinguishable from a fact — and the journal prompt asks the model to
    # be "evocative". That is the pipeline that puts a full moon and a prison
    # into her memory of an afternoon spent making a PDF about orcas.
    provenance = provenance_label(metadata_for_provenance)
    if provenance:
        attributes.append(f'provenance="{_sanitize_attribute(provenance)}"')
    if memory_type in {"fact", "preference", "recent_episode", "shared_ground"}:
        attributes.append(f'type="{memory_type}"')
    if source:
        attributes.append(f'source="{_sanitize_attribute(source)}"')
    opening = _RECALL_OPEN + ("" if not attributes else " " + " ".join(attributes)) + ">"
    return f"{opening}{text}{_RECALL_CLOSE}"


def _sanitize_attribute(value: str) -> str:
    """Keep an attribute from breaking out of the boundary it describes."""
    cleaned = _neutralize_role_markers(str(value or ""))
    for character in ('"', "<", ">", "\n", "\r"):
        cleaned = cleaned.replace(character, "")
    return cleaned.strip()[:80]


async def _call_memory_method(
    method: Any,
    *args: Any,
    timeout_s: float | None = None,
    **kwargs: Any,
) -> Any:
    if method is None:
        return None

    async def _invoke() -> Any:
        if inspect.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        result = await asyncio.to_thread(method, *args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    if timeout_s is None:
        return await _invoke()
    return await asyncio.wait_for(_invoke(), timeout=max(0.05, float(timeout_s)))


async def _hydrate_runtime_memory(payload_state: Any, objective: str) -> None:
    if payload_state is None or not objective:
        return

    snippets: list[str] = []
    seen: set[str] = set()
    timeout_s = _memory_hydration_timeout_s()

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

    async def _optional_memory_call(stage: str, method: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await _call_memory_method(method, *args, **kwargs)
        except TimeoutError as exc:
            logger.debug(
                "Runtime memory hydration source timed out; continuing with available state memory "
                "(stage=%s, objective=%r): %s",
                stage,
                objective[:120],
                exc,
            )
            return None
        except _SOFT_RUNTIME_FAILURES as exc:
            if os.environ.get("AURA_STRICT_RUNTIME_MEMORY_HYDRATION", "").strip() == "1":
                _record_runtime_wiring_degradation(
                    exc,
                    stage=f"runtime_memory_hydration.{stage}",
                    action="continued payload assembly after optional memory hydration source failed",
                    severity="warning",
                    extra={"objective_preview": objective[:160]},
                )
            else:
                logger.debug(
                    "Runtime memory hydration source failed; continuing with available state memory "
                    "(stage=%s, objective=%r): %s",
                    stage,
                    objective[:120],
                    exc,
                )
            return None

    try:
        memory = service_access.resolve_memory_facade(default=None)
        if memory is not None:
            search_method = getattr(memory, "search", None)
            if search_method is not None:
                for item in list(
                    await _optional_memory_call(
                        "memory_facade.search",
                        search_method,
                        objective,
                        limit=5,
                        timeout_s=timeout_s,
                    ) or []
                ):
                    _push(item)

            hot_method = getattr(memory, "get_hot_memory", None)
            if hot_method is not None:
                hot = await _optional_memory_call(
                    "memory_facade.hot",
                    hot_method,
                    limit=3,
                    timeout_s=min(timeout_s, 0.75),
                )
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
                    await _optional_memory_call(
                        "knowledge_graph.search",
                        search_knowledge,
                        objective,
                        limit=3,
                        timeout_s=timeout_s,
                    ) or []
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
    payload_state = runtime_state
    contract: ResponseContract | None = None
    prepared_messages = messages

    if runtime_state is not None and objective:
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
            # What she is attending to on THIS turn.
            #
            # The snapshot carried the objective and left attention_focus at
            # whatever a background loop had last written, so a reader asking
            # what the turn was about could get an answer from a different
            # one. The clone exists to be turn-consistent; a field that
            # survives the clone unchanged is the part that is not.
            payload_state.cognition.attention_focus = objective
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

        if not is_background and _should_hydrate_runtime_memory(objective, origin):
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
            # Not a bare None. See ContractUnavailable: this turn is being
            # generated WITHOUT its response contract, which is a different
            # fact from a turn that never needed one.
            contract = ContractUnavailable(
                reason="contract_construction_failed",
                error=f"{type(exc).__name__}: {exc}",
            )

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
        prompt, _inferred_system = _coerce_prompt_from_messages(prepared_messages)
        # Structured messages are authoritative. Returning the same system
        # content again as a scalar causes chat clients to prepend it a second
        # time at their transport boundary.
        system_prompt = None
    elif contract and contract.reason != "ordinary_dialogue":
        block = contract.to_prompt_block().strip()
        system_prompt = f"{system_prompt}\n\n{block}".strip() if system_prompt else block

    # Return the same derived snapshot used to build messages and the response
    # contract. Downstream sampler/voice overrides must not re-read the mutable
    # live state and describe a different moment than the model prompt.
    return str(prompt or ""), system_prompt, prepared_messages, contract, payload_state


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
        if hasattr(sve, "get_generation_params_for"):
            overrides = dict(
                sve.get_generation_params_for(
                    state=runtime_state,
                    user_message=str(objective or "")[:500],
                    origin=str(origin or "system"),
                )
                or {}
            )
        else:
            sve.compile_profile(
                state=runtime_state,
                user_message=str(objective or "")[:500],
                origin=str(origin or "system"),
            )
            overrides = dict(sve.get_generation_params() or {})
        profile = sve.get_current_profile()
        if overrides:
            source = str(getattr(profile, "compilation_source", "") or "substrate_voice")
            if overrides.get("substrate_profile_reused"):
                source = f"{source}, reused_runtime_profile"
            overrides["substrate_generation_source"] = source
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
        # NOT an empty dict. `{}` from a failed compilation was
        # indistinguishable from `{}` meaning "no overrides were needed", so
        # substrate-driven sampling and voice constraints vanished with
        # nothing the caller could see — the degradation went to the log and
        # the generation proceeded on caller defaults as though that had
        # been the intent.
        #
        # `substrate_generation_source` is already the field the success
        # path uses to say where the parameters came from, and the router
        # passes it through to the request, so saying "unavailable" here
        # reaches the receipt by the route that already exists. No sampling
        # VALUES are invented: the caller's defaults still apply, which is
        # the honest outcome, and now it is a stated one.
        return {
            "substrate_generation_source": (
                f"unavailable:{type(exc).__name__}"
            )
        }


def build_agentic_tool_map(
    required_skill: str | Sequence[str] | None = None,
    *,
    objective: str | None = None,
    max_tools: int = 8,
) -> dict[str, Any] | None:
    try:
        if required_skill is None and looks_like_capability_inventory_request(str(objective or "")):
            return None
        from core.container import ServiceContainer

        cap = ServiceContainer.get("capability_engine", default=None)
        if not cap or not hasattr(cap, "get_tool_definitions"):
            return None

        # `required_skill` may name one capability or the working set for the
        # turn. Filtering to exactly one made every multi-step task impossible
        # by construction: reading a file and then running what it says needs
        # two, and checking the result needs a third.
        if required_skill is None:
            wanted: set[str] = set()
        elif isinstance(required_skill, str):
            wanted = {required_skill}
        else:
            wanted = {str(item) for item in required_skill if str(item).strip()}

        # The registry's selector takes ONE name. Handed a set it matched
        # nothing and the turn was offered no tools at all — the working set
        # was derived correctly and then thrown away one call later. Ask it
        # for a wide slate and do the membership filtering here, which is
        # where the set is understood.
        if hasattr(cap, "select_tool_definitions"):
            tool_defs = (
                cap.select_tool_definitions(
                    objective=str(objective or ""),
                    required_skill=next(iter(wanted)) if len(wanted) == 1 else None,
                    max_tools=max(int(max_tools or 1), len(wanted) or 1, 8),
                    # The working set was already decided. Handing it over
                    # means the registry fetches those definitions by name
                    # instead of ranking the same objective a second time and
                    # intersecting the two answers — which is how build_app
                    # was wanted and never offered.
                    requested=sorted(wanted),
                )
                or []
            )
        else:
            tool_defs = cap.get_tool_definitions() or []

        # A named capability must survive the selector's own ranking: it was
        # asked for, so it cannot be dropped for being less relevant.
        if wanted and not {
            str((entry.get("function", {}) or {}).get("name"))
            for entry in tool_defs
            if isinstance(entry, dict)
        } >= wanted:
            for entry in cap.get_tool_definitions() or []:
                if not isinstance(entry, dict):
                    continue
                if str((entry.get("function", {}) or {}).get("name")) in wanted:
                    tool_defs.append(entry)

        tools: dict[str, Any] = {}
        for entry in tool_defs:
            fn = entry.get("function", {}) if isinstance(entry, dict) else {}
            name = fn.get("name")
            if not name:
                continue
            if wanted and str(name) not in wanted:
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


_TRUE_FLAG_VALUES = frozenset({"1", "true", "yes", "on", "enable", "enabled"})


def _env_flag_enabled(name: str) -> bool:
    """True only when an environment flag is explicitly switched ON.

    Presence is not truth: "0", "false", "no", "off" and "" all mean off.
    Anything unrecognised is treated as off, so a typo cannot silently
    disable a guard.
    """
    return str(os.environ.get(name, "")).strip().lower() in _TRUE_FLAG_VALUES


def should_force_tool_handoff(contract: ResponseContract | None, *, is_background: bool) -> bool:
    """Whether a turn requiring search must hand off to tools before answering.

    CP126 8eda805e. The embodied-challenge escape tested only that the
    environment variable was PRESENT and non-empty, so exporting it as "0",
    "false", "no" or "disabled" — every conventional way to turn a flag OFF
    — silently switched off a mandatory handoff and let the model answer a
    search-requiring question with no tool evidence. A flag whose "off"
    values mean "on" is worse than no flag.
    """
    if _env_flag_enabled("AURA_EMBODIED_CHALLENGE"):
        return False
    if contract and getattr(contract, "requires_capability_inventory", False):
        return False
    # `required_skill` was a general field holding one value: web_search. So
    # this whole handoff — and the tool-calling loop behind it, which parses a
    # call, binds it to the tool's advertised schema, executes it and feeds the
    # result back — was reachable for search and for nothing else. Asked to run
    # Python with code_repl READY, the model was handed no tool at all, wrote
    # an answer instead, and stated an invented "Output:". Sixty-odd other
    # capabilities sat behind the same gate.
    return bool(
        contract
        and (contract.requires_search or contract.required_skill)
        and not contract.tool_evidence_available
        and not is_background
    )
