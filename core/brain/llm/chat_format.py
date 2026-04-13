from __future__ import annotations

from typing import Dict, Iterable, List, Optional

_ROLE_ALIASES = {
    "assistant": "assistant",
    "system": "system",
    "tool": "user",
    "developer": "system",
    "model": "assistant",
}


def _normalize_role(role: Optional[str]) -> str:
    normalized = str(role or "user").strip().lower()
    return _ROLE_ALIASES.get(normalized, "user")


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

    for message in messages:
        content = str(message.get("content", "") or "").strip()
        if not content:
            continue
        role = _normalize_role(message.get("role"))
        prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

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
