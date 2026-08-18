from __future__ import annotations

import hashlib as _hashlib
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger("Aura.ChatFormat")

_ROLE_ALIASES = {
    "assistant": "assistant",
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


def _normalize_role(role: Optional[str]) -> str:
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
_THINKING_SUPPORT: Dict[str, bool] = {}
_THINKING_PROBE = [{"role": "user", "content": "probe"}]


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

    def _render(flag: bool) -> Optional[str]:
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


def render_chat_template(
    tokenizer: object,
    messages: object,
    *,
    tools: object = None,
    add_generation_prompt: bool = True,
    enable_thinking: Optional[bool] = None,
) -> str:
    """Render a chat template, applying reasoning control when supported.

    ``enable_thinking=None`` leaves the model's own default alone. Raises
    whatever the tokenizer raises — callers already distinguish a tool-schema
    failure (which must not degrade to prose) from a plain one.
    """
    apply = getattr(tokenizer, "apply_chat_template")
    kwargs = {
        "tools": tools,
        "add_generation_prompt": add_generation_prompt,
        "tokenize": False,
    }
    if enable_thinking is not None and template_supports_thinking(tokenizer):
        kwargs["enable_thinking"] = bool(enable_thinking)
    return str(apply(messages, **kwargs))


def render_chat_continuation_template(
    tokenizer: object,
    messages: object,
    *,
    tools: object = None,
    enable_thinking: Optional[bool] = None,
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

    apply = getattr(tokenizer, "apply_chat_template")
    kwargs = {
        "tools": tools,
        "add_generation_prompt": False,
        "tokenize": False,
        "continue_final_message": True,
    }
    if enable_thinking is not None and template_supports_thinking(tokenizer):
        kwargs["enable_thinking"] = bool(enable_thinking)
    try:
        rendered = str(apply(messages, **kwargs))
    except (TypeError, ValueError, KeyError, RuntimeError, AttributeError):
        fallback_kwargs = {
            "tools": tools,
            "add_generation_prompt": False,
            "tokenize": False,
        }
        if enable_thinking is not None and template_supports_thinking(tokenizer):
            fallback_kwargs["enable_thinking"] = bool(enable_thinking)
        rendered = str(apply(messages, **fallback_kwargs))
        prefix_end = rendered.rfind(partial)
        if prefix_end < 0:
            raise ValueError("chat template did not preserve the continuation prefix")
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


def thinking_enabled_for_model(model_name: Optional[str]) -> Optional[bool]:
    """Reasoning-mode policy per lane. None means 'use the model's default'.

    Only the fast lanes are pinned. The Brainstem (background/reflex) and the
    1.5B Reflex fallback exist to answer quickly inside small budgets; the
    Cortex and Solver are where deliberation is the point, and they keep
    whatever the artifact ships with.
    """
    name = str(model_name or "").strip().lower()
    if not name:
        return None
    if "brainstem" in name or "9b" in name or "1.5b" in name or "7b" in name:
        return False
    return None


def _uses_grok_chat_template(model_name: Optional[str]) -> bool:
    return "grok" in str(model_name or "").strip().lower()


def _format_grok_messages(
    messages: Iterable[Dict[str, str]],
    *,
    require_json: bool = False,
) -> str:
    prompt_parts: List[str] = []

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
    messages: Iterable[Dict[str, str]],
    *,
    require_json: bool = False,
    model_name: Optional[str] = None,
) -> str:
    """Serialize messages using the ChatML/Qwen instruct format."""
    if _uses_grok_chat_template(model_name):
        return _format_grok_messages(messages, require_json=require_json)

    prompt_parts: List[str] = []
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
    system_prompt: Optional[str] = None,
    *,
    model_name: Optional[str] = None,
) -> str:
    messages: List[Dict[str, str]] = []
    if system_prompt and str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": str(prompt or "")})
    return format_chatml_messages(messages, model_name=model_name)
