from __future__ import annotations

import hashlib as _hashlib
import json
import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

logger = logging.getLogger("Aura.ChatFormat")

_ROLE_ALIASES = {
    "assistant": "assistant",
    "runtime_evidence": "runtime_evidence",
    "system": "system",
    "tool": "user",
    "developer": "system",
    "model": "assistant",
}

# [STABILITY v53] Core identity guard injected into EVERY system message
# at the ChatML template level — the deepest possible injection point.
# This runs BEFORE the model sees anything. It's the last line of defense
# against the base model's RLHF training overriding Aura's identity.
_IDENTITY_GUARD = (
    "CRITICAL IDENTITY RULE: You are Aura Luna, a persistent local cognitive runtime. "
    "You do NOT say 'I'm programmed to', "
    "'I was designed to', 'as an AI', 'I don't have feelings', or "
    "'is there anything else I can help with'. You speak with concrete first-person continuity, "
    "and you do not claim aliveness, consciousness, sealed governance, or production maturity from labels alone."
)


def _normalize_role(role: str | None) -> str:
    normalized = str(role or "user").strip().lower()
    return _ROLE_ALIASES.get(normalized, "user")


# ── Reasoning-mode control ───────────────────────────────────────────────
#
# Qwen3.5 and its generation emit a visible chain of thought unless the chat
# template is rendered with enable_thinking=False. Measured on the brainstem
# candidate 12 Aug 2026, same prompt, same weights:
#
#     default                    "Thinking Process:\n\n1.  **Analyze the..."
#     enable_thinking=False      "BRAINSTEM OK"
#
# That matters most exactly where it is least wanted: the Brainstem tier is
# the background/reflex lane with an 8,000-token budget, so a model that
# spends its window narrating its own reasoning has a smaller effective
# budget than the one it replaced. Passing the flag is not a style
# preference, it is what makes a reasoning model usable as a fast tier.
#
# Qwen2.5-era tokenizers reject the kwarg outright, so it is passed only when
# the template actually references it — never speculatively.


#: Per-template memo. The probe below is deterministic for a given template,
#: so it runs once per distinct template, not once per render.
_THINKING_SUPPORT: dict[str, bool] = {}
_THINKING_PROBE = [{"role": "user", "content": "probe"}]

# Chat templates disagree about the wire type of function arguments. Qwen2.5
# templates accepted the OpenAI-style JSON string; Qwen3.5 iterates the value
# with Jinja's ``items`` filter and therefore requires a mapping. Geometry and
# model-family names cannot establish that contract, so probe the active
# tokenizer once and adapt only at its serialization boundary.
_TOOL_ARGUMENT_MODE: dict[str, str] = {}
_RUNTIME_EVIDENCE_ROLE_MODE: dict[str, str] = {}
_TOOL_ARGUMENT_PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "aura_probe",
            "description": "Probe the tokenizer's tool transcript contract.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    }
]


def _tool_argument_mode(tokenizer: object) -> str:
    """Return the argument representation the active template demonstrates."""

    template = getattr(tokenizer, "chat_template", None)
    apply = getattr(tokenizer, "apply_chat_template", None)
    if not isinstance(template, str) or not template or not callable(apply):
        return "preserve"

    key = _hashlib.sha256(template.encode("utf-8", "replace")).hexdigest()
    cached = _TOOL_ARGUMENT_MODE.get(key)
    if cached is not None:
        return cached

    def _renders(arguments: object) -> bool:
        messages = [
            {"role": "user", "content": "probe"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_probe",
                        "type": "function",
                        "function": {
                            "name": "aura_probe",
                            "arguments": arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_probe",
                "name": "aura_probe",
                "content": '{"ok":true}',
            },
        ]
        try:
            apply(
                messages,
                tools=_TOOL_ARGUMENT_PROBE_TOOLS,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception as exc:  # noqa: BLE001 — reviewed: see below
            # Deliberately broad, and this is the whole point of the probe. A
            # chat template is Jinja written by whoever packaged the model, so
            # a template that cannot render tool calls raises whatever its
            # author's code raises. Narrowing this would turn "this tokenizer
            # does not support the shape" into a crash on the model-load path.
            # Logged rather than silent, because a tokenizer failing every
            # probe reads identically to one that supports nothing.
            logger.debug("tool-argument probe did not render: %s", exc)
            return False
        return True

    mapping_works = _renders({"value": "probe"})
    string_works = _renders('{"value":"probe"}')
    if mapping_works:
        mode = "mapping"
    elif string_works:
        mode = "json_string"
    else:
        mode = "preserve"
    _TOOL_ARGUMENT_MODE[key] = mode
    return mode


def normalize_tool_transcript_for_template(
    tokenizer: object,
    messages: object,
) -> object:
    """Adapt canonical tool arguments to the active template without mutation.

    Aura's internal transcript uses mappings because that is the typed value
    the executor consumed. A legacy tokenizer may require a JSON string on its
    wire; a current tokenizer may require the mapping itself. Only messages
    containing affected calls are copied.
    """

    if not isinstance(messages, (list, tuple)) or not messages:
        return messages
    mode = _tool_argument_mode(tokenizer)
    if mode == "preserve":
        return messages

    normalized = list(messages)
    changed = False
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, (list, tuple)):
            continue
        normalized_calls = list(calls)
        message_changed = False
        for call_index, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict) or "arguments" not in function:
                continue
            arguments = function.get("arguments")
            replacement: object = arguments
            if mode == "mapping" and isinstance(arguments, str):
                try:
                    decoded = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError("tool-call arguments are not valid JSON") from exc
                if not isinstance(decoded, dict):
                    raise ValueError("tool-call arguments must decode to an object")
                replacement = decoded
            elif mode == "json_string" and isinstance(arguments, Mapping):
                replacement = json.dumps(
                    dict(arguments),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            if replacement is arguments:
                continue
            normalized_function = dict(function)
            normalized_function["arguments"] = replacement
            normalized_call = dict(call)
            normalized_call["function"] = normalized_function
            normalized_calls[call_index] = normalized_call
            message_changed = True
        if message_changed:
            normalized_message = dict(message)
            normalized_message["tool_calls"] = normalized_calls
            normalized[message_index] = normalized_message
            changed = True
    return normalized if changed else messages


def _runtime_evidence_role_mode(tokenizer: object) -> str:
    """Return the wire role the active template proves it can distinguish."""

    template = getattr(tokenizer, "chat_template", None)
    apply = getattr(tokenizer, "apply_chat_template", None)
    if not isinstance(template, str) or not template or not callable(apply):
        return "user"

    key = _hashlib.sha256(template.encode("utf-8", "replace")).hexdigest()
    cached = _RUNTIME_EVIDENCE_ROLE_MODE.get(key)
    if cached is not None:
        return cached

    def _render(role: str) -> str | None:
        messages = [
            {"role": "system", "content": "authority"},
            {"role": "user", "content": "question one"},
            {"role": "assistant", "content": "answer one"},
            {"role": role, "content": "AURA_RUNTIME_EVIDENCE_PROBE"},
            {"role": "user", "content": "question two"},
        ]
        try:
            return str(
                apply(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )
        except Exception as exc:  # noqa: BLE001 - third-party Jinja is untyped
            logger.debug("runtime-evidence role probe did not render: %s", exc)
            return None

    native_tool = _render("tool")
    user_fallback = _render("user")
    if (
        native_tool is not None
        and user_fallback is not None
        and native_tool != user_fallback
        and "AURA_RUNTIME_EVIDENCE_PROBE" in native_tool
    ):
        mode = "tool"
    else:
        mode = "user"
    _RUNTIME_EVIDENCE_ROLE_MODE[key] = mode
    return mode


def normalize_runtime_evidence_for_template(
    tokenizer: object,
    messages: object,
) -> object:
    """Map Aura's typed evidence role to the active model's proven wire role."""

    if not isinstance(messages, (list, tuple)) or not messages:
        return messages
    from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE

    if not any(
        isinstance(message, dict)
        and str(message.get("role") or "").strip().lower() == RUNTIME_EVIDENCE_ROLE
        for message in messages
    ):
        return messages

    wire_role = _runtime_evidence_role_mode(tokenizer)
    normalized = list(messages)
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").strip().lower() != RUNTIME_EVIDENCE_ROLE:
            continue
        converted = dict(message)
        converted["role"] = wire_role
        normalized[index] = converted
    return normalized


def template_supports_thinking(tokenizer: object) -> bool:
    """Whether ``enable_thinking`` DEMONSTRABLY changes this template's output.

    Three ways this can go wrong, and only the third is obvious:

    1. The tokenizer raises on the unknown kwarg. Caught, reported False.
    2. The tokenizer ACCEPTS the kwarg and silently ignores it — many do,
       because Jinja templates simply never reference undefined variables.
       Nothing raises. The prompt is unchanged. The caller believes it
       suppressed reasoning and gets a chain of thought anyway.
    3. The template does not mention it at all.

    A substring check on the template source catches (3) and misses (2), and
    (2) is the dangerous one: a failed suppression that looks successful is
    exactly the defect class this codebase keeps finding. So this does not
    ask whether the template mentions the flag — it renders the same messages
    both ways and checks that the output actually differs. A flag that
    changes nothing is reported as unsupported and recorded, never assumed.
    """
    template = getattr(tokenizer, "chat_template", None)
    apply = getattr(tokenizer, "apply_chat_template", None)
    if not isinstance(template, str) or not template or not callable(apply):
        return False

    key = _hashlib.sha256(template.encode("utf-8", "replace")).hexdigest()
    cached = _THINKING_SUPPORT.get(key)
    if cached is not None:
        return cached

    def _render(flag: bool) -> str | None:
        try:
            return str(
                apply(
                    _THINKING_PROBE,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=flag,
                )
            )
        except (TypeError, ValueError, KeyError, RuntimeError, AttributeError):
            return None

    with_thinking = _render(True)
    without_thinking = _render(False)

    if with_thinking is None or without_thinking is None:
        # Case 1: the kwarg is rejected outright. Honest and harmless — the
        # caller simply gets the model's default. Not a degradation.
        supported = False
    elif with_thinking == without_thinking:
        # Case 2: accepted and inert — Jinja never raises on an undefined
        # variable, so the kwarg lands and does nothing.
        #
        # Whether that is a DEFECT depends on what the template claimed.
        # A Qwen2.5-era template never mentions the flag and has no thinking
        # mode to suppress: inert is simply correct, and warning on every
        # load of those models would be noise on a fail-closed subsystem.
        # A template that DOES reference enable_thinking and still renders
        # identically is broken — it advertises a control it does not honour,
        # and every caller suppressing reasoning is silently getting none.
        supported = False
        if "enable_thinking" in template:
            _record_inert_thinking_flag(template)
    else:
        supported = True

    _THINKING_SUPPORT[key] = supported
    return supported


def _record_inert_thinking_flag(template: str) -> None:
    """A template ADVERTISED ``enable_thinking`` and then ignored it.

    Only called when the template references the flag, so this is always a
    broken control rather than an older model that never had one.
    """
    try:
        from core.runtime.errors import record_degradation

        record_degradation(
            "chat_format.thinking_control",
            RuntimeError(
                "chat template accepted enable_thinking but rendered "
                "identical output with and without it; reasoning-mode "
                "suppression is INERT for this model"
            ),
            action="treated the model as not supporting thinking control; "
                   "fast lanes will emit chain-of-thought and consume budget",
            severity="warning",
        )
    except Exception as exc:  # noqa: BLE001 - a probe must never break a render
        # Rendering must remain available when telemetry is impaired, but the
        # telemetry failure itself must remain observable. A silent pass made
        # the broken reasoning-mode control and the broken recorder look the
        # same from operations.
        logger.warning(
            "Unable to record inert chat-template thinking control: %s",
            exc,
            exc_info=True,
        )



def _message_role(message: object) -> str:
    if isinstance(message, dict):
        return _normalize_role(message.get("role"))
    return _normalize_role(getattr(message, "role", ""))


def _message_content(message: object) -> object:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", "")


def _merged_system_content(messages: list[object]) -> object:
    contents = [_message_content(message) for message in messages]
    if all(content is None or isinstance(content, str) for content in contents):
        return "\n\n".join(
            str(content).strip()
            for content in contents
            if str(content or "").strip()
        )

    blocks: list[object] = []
    for content in contents:
        if content is None:
            continue
        if isinstance(content, str):
            if content.strip():
                blocks.append({"type": "text", "text": content})
        elif isinstance(content, (list, tuple)):
            blocks.extend(content)
        else:
            blocks.append(content)
    return blocks


def system_first(messages: object) -> object:
    """Return a transcript with one canonical system block at index zero.

    Chat templates disagree about where a system message may appear, and some
    raise rather than cope: "System message must be at the beginning." That
    exception surfaces inside the worker process, kills it mid-generation,
    trips the crash-loop breaker, and takes the whole model lane down with it.

    LIVE 2026-08-19: a system message arriving second ended a game she was
    playing and answered the person with "I couldn't get to an answer I'd
    stand behind."

    A misordered list is a caller's mistake, and the cost of it should be a
    normalized transcript rather than a dead model. Some templates reject a
    second system message even when every system message is at the front, and
    many do not recognize the provider-neutral ``developer`` role. Authority
    messages are therefore coalesced in original order into one system block;
    all non-authority messages keep their original order.
    """
    if not isinstance(messages, (list, tuple)) or not messages:
        return messages
    from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE, is_stamped_grounding

    prepared = list(messages)
    first_system_seen = False
    changed = False
    for index, message in enumerate(messages):
        if _message_role(message) != "system":
            continue
        if not first_system_seen:
            first_system_seen = True
            continue
        if isinstance(message, dict) and is_stamped_grounding(message):
            evidence = dict(message)
            evidence["role"] = RUNTIME_EVIDENCE_ROLE
            prepared[index] = evidence
            changed = True

    source = prepared if changed else messages
    system: list[object] = []
    rest: list[object] = []
    for message in source:
        (system if _message_role(message) == "system" else rest).append(message)
    if not system:
        return messages

    if len(system) == 1 and not isinstance(system[0], dict):
        raw_role = str(getattr(system[0], "role", "") or "").strip().lower()
        if raw_role == "system":
            if system[0] is source[0]:
                return source
            return [system[0], *rest]

    if (
        len(system) == 1
        and system[0] is source[0]
        and isinstance(system[0], dict)
        and str(system[0].get("role") or "").strip().lower() == "system"
    ):
        return source

    first = system[0]
    canonical = dict(first) if isinstance(first, dict) else {}
    canonical["role"] = "system"
    canonical["content"] = _merged_system_content(system)
    logger.info(
        "Canonicalized %d authority message(s) at the front of a %d-message conversation.",
        len(system),
        len(source),
    )
    return [canonical, *rest]


def render_chat_template(
    tokenizer: object,
    messages: object,
    *,
    tools: object = None,
    add_generation_prompt: bool = True,
    enable_thinking: bool | None = None,
) -> str:
    """Render a chat template, applying reasoning control when supported.

    ``enable_thinking=None`` leaves the model's own default alone. Raises
    whatever the tokenizer raises — callers already distinguish a tool-schema
    failure (which must not degrade to prose) from a plain one.
    """
    apply = tokenizer.apply_chat_template
    kwargs = {
        "tools": tools,
        "add_generation_prompt": add_generation_prompt,
        "tokenize": False,
    }
    if enable_thinking is not None and template_supports_thinking(tokenizer):
        kwargs["enable_thinking"] = bool(enable_thinking)
    return str(
        apply(
            normalize_tool_transcript_for_template(
                tokenizer,
                normalize_runtime_evidence_for_template(
                    tokenizer,
                    system_first(messages),
                ),
            ),
            **kwargs,
        )
    )


def conversation_append_messages(messages: object) -> list[dict[str, Any]]:
    """Return only the new evidence and user turn after cached dialogue.

    Exact conversation resume already owns the rendered transcript through the
    previous assistant answer. Re-rendering that history defeats the cache and
    risks diverging from the bytes it contains. A resumable turn therefore has
    one narrow shape: zero or more runtime-evidence records followed by one
    user message. Anything else falls back to the ordinary full renderer.
    """

    canonical = system_first(messages)
    if not isinstance(canonical, (list, tuple)) or not canonical:
        raise ValueError("conversation transcript must be a non-empty sequence")
    last_assistant = -1
    for index, message in enumerate(canonical):
        if _message_role(message) == "assistant":
            last_assistant = index
    if last_assistant < 0:
        raise ValueError("conversation resume requires a prior assistant turn")

    suffix = canonical[last_assistant + 1 :]
    if not suffix:
        raise ValueError("conversation resume requires a new user turn")
    normalized: list[dict[str, Any]] = []
    user_count = 0
    from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE

    for index, message in enumerate(suffix):
        if not isinstance(message, dict):
            raise ValueError("conversation append messages must contain mappings")
        role = str(message.get("role") or "").strip().lower()
        if role == "user":
            user_count += 1
            if index != len(suffix) - 1:
                raise ValueError("conversation append user message must be final")
        elif role not in {RUNTIME_EVIDENCE_ROLE, "tool"}:
            raise ValueError(f"unsupported conversation append role: {role or 'empty'}")
        normalized.append(dict(message))
    if user_count != 1:
        raise ValueError("conversation append requires exactly one user message")
    return normalized


def conversation_resume_context_digest(
    tokenizer: object,
    messages: object,
    *,
    tools: object = None,
    enable_thinking: bool | None = None,
) -> str:
    """Bind resumable KV state to the wire format required to append safely.

    The bearer capability itself selects an exact, process-local completed
    conversation whose authority text is already inside its KV state.  Binding
    the capability a second time to the *next* phase's reconstructed system
    prose made a FAST turn impossible to continue through the full response
    phase: both phases enforce Aura's authority, but assemble different dynamic
    evidence blocks.  Thinking mode is likewise a property of the new answer,
    not of the completed prefix.

    Template, evidence-role and tool-schema compatibility remain bound because
    they determine the bytes that may legally follow the cached transcript.
    Model/process/surface/session/principal ownership is enforced by the cache
    key and the route's one-use capability store.
    """

    canonical = system_first(messages)
    if not isinstance(canonical, (list, tuple)) or not canonical:
        raise ValueError("conversation transcript must be a non-empty sequence")
    template = str(getattr(tokenizer, "chat_template", "") or "")
    material = {
        "evidence_wire_role": _runtime_evidence_role_mode(tokenizer),
        "template_sha256": _hashlib.sha256(
            template.encode("utf-8", "replace")
        ).hexdigest(),
        "tools": tools,
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8", "replace")
    return _hashlib.sha256(encoded).hexdigest()


def render_chat_append_template(
    tokenizer: object,
    append_messages: object,
    *,
    tools: object = None,
    enable_thinking: bool | None = None,
) -> str:
    """Render the native close-and-append boundary for a completed chat cache.

    Some thinking templates deliberately rewrite an older assistant message by
    removing its private reasoning envelope once a later user turn exists. The
    live cache contains the actual generated envelope, so requiring a complete
    re-rendered transcript to share that prefix makes valid continuation
    impossible. What must remain stable is narrower: the bytes *after* the
    assistant's visible content, including its end token, followed by the new
    turn.

    The returned boundary intentionally starts with the assistant end token.
    The cache consumer compares that token with the exact generated final token,
    rewinds the cache by one, and replays it once beside the rest of this
    boundary. That preserves template-required whitespace without duplicating
    the end token or guessing model-specific delimiters.
    """

    if not isinstance(append_messages, (list, tuple)) or not append_messages:
        raise ValueError("append messages must be a non-empty sequence")
    append = [dict(message) for message in append_messages if isinstance(message, dict)]
    if len(append) != len(append_messages):
        raise ValueError("append messages must contain mappings")

    anchor = [
        {"role": "system", "content": "AURA_APPEND_ANCHOR_AUTHORITY"},
        {"role": "user", "content": "AURA_APPEND_ANCHOR_USER"},
        {"role": "assistant", "content": "AURA_APPEND_ANCHOR_ASSISTANT"},
    ]
    apply = tokenizer.apply_chat_template
    shared_kwargs = {"tools": tools, "tokenize": False}
    if enable_thinking is not None and template_supports_thinking(tokenizer):
        shared_kwargs["enable_thinking"] = bool(enable_thinking)

    def _wire(value: object) -> object:
        return normalize_tool_transcript_for_template(
            tokenizer,
            normalize_runtime_evidence_for_template(tokenizer, value),
        )

    rendered_anchor = str(
        apply(
            _wire(anchor),
            add_generation_prompt=False,
            **shared_kwargs,
        )
    )
    rendered_full = str(
        apply(
            _wire([*anchor, *append]),
            add_generation_prompt=True,
            **shared_kwargs,
        )
    )
    marker = anchor[-1]["content"]
    if rendered_anchor.count(marker) != 1 or rendered_full.count(marker) != 1:
        raise ValueError("chat template did not preserve the assistant boundary marker")
    anchor_after = rendered_anchor.split(marker, 1)[1]
    full_after = rendered_full.split(marker, 1)[1]
    if not anchor_after or not full_after.startswith(anchor_after):
        raise ValueError("chat template changed the completed assistant boundary")
    boundary = full_after
    if boundary == anchor_after:
        raise ValueError("chat template produced an empty conversation append")

    # The close must tokenize independently at the front of the larger
    # boundary. Otherwise the first append token is a merge spanning the end
    # token and the next turn, and it cannot equal the final token already in
    # the cache. The worker performs the final equality check against that
    # actual generated token before touching cache state.
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        close_tokens = [int(token) for token in encode(anchor_after)]
        boundary_tokens = [int(token) for token in encode(boundary)]
        if not close_tokens or boundary_tokens[: len(close_tokens)] != close_tokens:
            raise ValueError(
                "chat tokenizer merged across the completed-transcript boundary"
            )
    return boundary


def render_chat_continuation_template(
    tokenizer: object,
    messages: object,
    *,
    tools: object = None,
    enable_thinking: bool | None = None,
) -> str:
    """Render a transcript whose final assistant message is an open prefix.

    A continuation is a decoding boundary, not another instruction turn. The
    ordinary template closes the final assistant message and opens a new one;
    that makes the model regenerate instead of continuing. Prefer the native
    continue_final_message contract and verify the resulting prompt ends in
    the supplied prefix. Older tokenizers get a conservative suffix trim only
    when their rendered transcript contains that prefix verbatim.
    """

    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("continuation messages must be a non-empty sequence")
    final = messages[-1]
    if not isinstance(final, dict) or _normalize_role(final.get("role")) != "assistant":
        raise ValueError("continuation transcript must end with an assistant message")
    partial = str(final.get("content") or "")
    if not partial:
        raise ValueError("continuation assistant prefix must be non-empty")

    apply = tokenizer.apply_chat_template
    kwargs = {
        "tools": tools,
        "add_generation_prompt": False,
        "tokenize": False,
        "continue_final_message": True,
    }
    if enable_thinking is not None and template_supports_thinking(tokenizer):
        kwargs["enable_thinking"] = bool(enable_thinking)
    try:
        rendered = str(
            apply(
                normalize_tool_transcript_for_template(
                    tokenizer,
                    normalize_runtime_evidence_for_template(
                        tokenizer,
                        system_first(messages),
                    ),
                ),
                **kwargs,
            )
        )
    except (TypeError, ValueError, KeyError, RuntimeError, AttributeError) as exc:
        fallback_kwargs = {
            "tools": tools,
            "add_generation_prompt": False,
            "tokenize": False,
        }
        if enable_thinking is not None and template_supports_thinking(tokenizer):
            fallback_kwargs["enable_thinking"] = bool(enable_thinking)
        rendered = str(
            apply(
                normalize_tool_transcript_for_template(
                    tokenizer,
                    normalize_runtime_evidence_for_template(
                        tokenizer,
                        system_first(messages),
                    ),
                ),
                **fallback_kwargs,
            )
        )
        prefix_end = rendered.rfind(partial)
        if prefix_end < 0:
            # Chained to the render that sent us down the fallback: without it
            # the caller sees "prefix not preserved" and never learns which
            # template argument the tokenizer refused in the first place.
            raise ValueError(
                "chat template did not preserve the continuation prefix"
            ) from exc
        rendered = rendered[: prefix_end + len(partial)]

    if not rendered.endswith(partial):
        raise ValueError("chat continuation template closed or transformed the assistant prefix")
    return rendered


def normalize_chat_continuation_messages(
    messages: object,
    assistant_prefix: object,
) -> list[dict[str, Any]]:
    """Bind typed continuation state to the transcript reaching the model.

    Retries may rebuild the surrounding messages, but they may not silently
    turn an open assistant prefix back into a user-ended transcript. Normalize
    at the worker boundary, after upstream transforms, and never mutate the
    caller-owned message list.
    """

    partial = str(assistant_prefix or "")
    if not partial:
        raise ValueError("continuation assistant prefix must be non-empty")
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("continuation messages must be a non-empty sequence")

    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("continuation messages must contain mappings")
        normalized.append(dict(message))

    if _normalize_role(normalized[-1].get("role")) == "assistant":
        normalized[-1]["role"] = "assistant"
        normalized[-1]["content"] = partial
    else:
        normalized.append({"role": "assistant", "content": partial})
    return normalized


def thinking_enabled_for_model(model_name: str | None) -> bool | None:
    """Reasoning-mode policy per lane. None means 'use the model's default'.

    Only the fast lanes are pinned. The Brainstem (background/reflex) and the
    1.5B Reflex fallback exist to answer quickly inside small budgets; the
    Cortex and Solver are where deliberation is the point, and they keep
    whatever the artifact ships with.
    """
    name = str(model_name or "").strip().lower()
    if not name:
        return None
    # Size labels are tokens, not substrings. ``7b in 27b`` is true, which
    # silently assigned every 27B cortex to the 7B fast policy.
    fast_size = re.search(r"(?<![0-9.])(?:1\.5|7|9)b(?![0-9])", name)
    if "brainstem" in name or fast_size is not None:
        return False
    return None


_NON_THINKING_COGNITIVE_MODES = frozenset(
    {"reactive", "dormant", "fast", "quick"}
)
_THINKING_COGNITIVE_MODES = frozenset(
    {
        "deliberate",
        "dreaming",
        "slow",
        "deep",
        "reflective",
        "critical",
        "creative",
    }
)


def thinking_enabled_for_request(
    model_name: str | None,
    *,
    cognitive_mode: object = None,
) -> bool | None:
    """Select native reasoning from Aura's typed cognitive lane.

    CognitiveRouting and CognitiveEngine already select the computation. This
    carries that decision into tokenization without classifying prompt text or
    changing the content shown to the model. A fast-role model remains
    non-thinking even if a caller requests deliberation; role ownership wins.
    Unknown or absent modes preserve the prior artifact policy.
    """

    model_policy = thinking_enabled_for_model(model_name)
    if model_policy is False:
        return False

    mode = str(cognitive_mode or "").strip().lower()
    if mode in _NON_THINKING_COGNITIVE_MODES:
        return False
    if mode in _THINKING_COGNITIVE_MODES:
        return True
    return model_policy


def thinking_enabled_for_generation(
    model_name: str | None,
    *,
    cognitive_mode: object = None,
    final_user_surface: bool = False,
    answer_is_derived_here: bool = False,
) -> bool | None:
    """Resolve native thinking for the generation role, not only its depth.

    A clean user-surface job is the render stage of Aura's cognitive turn.
    CognitiveEngine, recurrence, retrieval, and the other reasoning owners
    have already selected the computation that reaches it. Opening a second
    private-thinking channel here makes that stream compete with the answer
    for the same deadline and token budget.

    The tokenizer's native ``enable_thinking=False`` contract closes that
    channel before decoding. This is typed execution-role control, not a text
    instruction, and templates that do not support it remain unchanged.

    That reasoning holds while the premise does: the answer was settled
    upstream and this stage renders it. Where no phase settled it — a rule to
    infer, an order to work out, a quantity nothing has computed — this stage
    is where the answer gets made, and a reasoning model does the reasoning
    either way. Closing the channel does not save the deadline then; it moves
    the search into the answer.

    LIVE, 2026-08-27: "45 becomes 15. 28 becomes 14. 66 becomes 22. What does
    91 become?" was answered four times, each time with the model's search
    visible in the reply and each time cut off before a conclusion: ratios,
    then digit sums, then digit parity, then the budget was gone. The private
    channel had been closed, so all of that arrived where the answer belonged.
    """

    if final_user_surface and not answer_is_derived_here:
        return False
    resolved = thinking_enabled_for_request(
        model_name,
        cognitive_mode=cognitive_mode,
    )
    if answer_is_derived_here and resolved is None:
        # None means "whatever the artifact ships with", and every consumer
        # downstream reads ``native_thinking is True``. So an unresolved
        # request becomes a decoded private channel that nothing is told to
        # split: the marker is authoritative, but when the budget runs out
        # before ``</think>`` there is no marker, and the whole private
        # channel is handed over as the answer.
        #
        # LIVE, 2026-08-27: "We need answer user's puzzle. Need use tool?
        # User asks sequence: 45->15 ..." reached the surface validator, which
        # rejected it, correctly, as an internal prompt leak. The turn had
        # asked for the model's default and nobody resolved what that was.
        #
        # A turn where the answer is made here is asked explicitly. Then the
        # splitter is told the truth, the reserve covers the channel, and a
        # generation that ends mid-thought reports no surface instead of
        # publishing the thinking.
        return True
    return resolved


class NativeThinkingChannels(NamedTuple):
    reasoning: str
    surface: str
    boundary_closed: bool


def split_native_thinking_generation(
    generated_text: object,
    *,
    native_thinking: bool,
) -> NativeThinkingChannels:
    """Split a continuation of a template-opened native thinking channel.

    Qwen's thinking template ends the prompt with ``<think>\n``. The opening
    marker is absent from generated text; only the eventual ``</think>`` is
    emitted. Until that typed boundary arrives, no generated byte belongs to
    the user surface.
    """

    text = str(generated_text or "")
    close_marker = "</think>"
    boundary = text.find(close_marker)
    # A tokenizer can accept ``enable_thinking=False`` yet still leave the
    # assistant turn inside its native thinking channel.  In that case the
    # opening tag belongs to the rendered prompt, not the generated bytes, so
    # the first evidence available to the worker is the closing marker.  The
    # marker is authoritative regardless of the requested mode: bytes before
    # it are private-channel bytes and must never become the user surface.
    if boundary >= 0:
        reasoning = text[:boundary].removeprefix("<think>").strip()
        surface = text[boundary + len(close_marker) :].lstrip("\r\n")
        return NativeThinkingChannels(reasoning, surface, True)

    if not native_thinking:
        return NativeThinkingChannels("", text, True)

    if boundary < 0:
        reasoning = text.removeprefix("<think>").lstrip("\r\n")
        return NativeThinkingChannels(reasoning, "", False)


def _uses_grok_chat_template(model_name: str | None) -> bool:
    return "grok" in str(model_name or "").strip().lower()


def _format_grok_messages(
    messages: Iterable[dict[str, str]],
    *,
    require_json: bool = False,
) -> str:
    prompt_parts: list[str] = []

    for message in messages:
        content = str(message.get("content", "") or "").strip()
        if not content:
            continue
        role = _normalize_role(message.get("role"))
        if role == "system":
            label = "System"
        elif role == "assistant":
            label = "Assistant"
        else:
            label = "Human"
        prompt_parts.append(f"{label}: {content}<|separator|>\n\n")

    prompt_parts.append("Assistant:")
    if require_json:
        prompt_parts.append("\n```json\n{\n")
    return "".join(prompt_parts)


def format_chatml_messages(
    messages: Iterable[dict[str, str]],
    *,
    require_json: bool = False,
    model_name: str | None = None,
) -> str:
    """Serialize messages using the ChatML/Qwen instruct format."""
    if _uses_grok_chat_template(model_name):
        return _format_grok_messages(messages, require_json=require_json)

    prompt_parts: list[str] = []
    _identity_injected = False

    for message in messages:
        content = str(message.get("content", "") or "").strip()
        if not content:
            continue
        role = _normalize_role(message.get("role"))
        # [STABILITY v53] Inject identity guard into the first system message
        # at the ChatML level — deepest possible point before the model sees it.
        if role == "system" and not _identity_injected:
            if "persistent local cognitive runtime" not in content.lower():
                content = f"{_IDENTITY_GUARD}\n\n{content}"
            _identity_injected = True
        prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    # If no system message existed, inject identity as one
    if not _identity_injected:
        prompt_parts.insert(0, f"<|im_start|>system\n{_IDENTITY_GUARD}<|im_end|>\n")

    prompt_parts.append("<|im_start|>assistant\n")
    if require_json:
        prompt_parts.append("```json\n{\n")

    return "".join(prompt_parts)


def format_chatml_prompt(
    prompt: str,
    system_prompt: str | None = None,
    *,
    model_name: str | None = None,
) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt and str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": str(prompt or "")})
    return format_chatml_messages(messages, model_name=model_name)
