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


def format_chatml_messages(
    messages: Iterable[Dict[str, str]],
    *,
    require_json: bool = False,
) -> str:
    """Serialize messages using the ChatML/Qwen instruct format."""
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


def format_chatml_prompt(prompt: str, system_prompt: Optional[str] = None) -> str:
    messages: List[Dict[str, str]] = []
    if system_prompt and str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": str(prompt or "")})
    return format_chatml_messages(messages)
