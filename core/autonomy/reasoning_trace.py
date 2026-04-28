"""core/autonomy/reasoning_trace.py
────────────────────────────────────
Reasoning-trace wrapper for LLM calls. Handles ``<think>...</think>`` blocks
(R1-style chain-of-thought) when the model emits them; degrades gracefully
when the model doesn't.

Why this exists
---------------
- Mythos / R1 / o1-class models emit visible reasoning blocks before
  their final answer. Aura's substrate can read those blocks to evaluate
  *how* the model reached its answer, not just what it said.
- Today's Qwen-32B-personality fuse doesn't emit `<think>` blocks. The
  wrapper falls back to treating the entire response as the "answer."
- When (a) a reasoning-trained base model is swapped in, or (b) the
  comprehension prompt explicitly asks for reasoning, this wrapper
  immediately starts producing rich traces without other code changing.

Public API:
    parsed = parse_reasoning_response(raw_text)
    parsed.thinking, parsed.answer, parsed.has_trace, parsed.token_estimate
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

# Match common reasoning-block delimiters used by various models
_THINK_BLOCK_PATTERNS = [
    re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<thinking>(.*?)</thinking>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<reasoning>(.*?)</reasoning>", re.IGNORECASE | re.DOTALL),
    re.compile(r"\[REASONING\](.*?)\[/REASONING\]", re.IGNORECASE | re.DOTALL),
]

# Open-ended (model emitted opening tag but truncated before close)
_THINK_OPEN_TAILS = [
    re.compile(r"<think>(.*)$", re.IGNORECASE | re.DOTALL),
    re.compile(r"<thinking>(.*)$", re.IGNORECASE | re.DOTALL),
]


@dataclass(frozen=True)
class ParsedResponse:
    raw: str
    thinking: Optional[str]   # None if no trace
    answer: str
    has_trace: bool
    truncated_trace: bool     # opened but did not close
    token_estimate: int       # rough token count of the answer


def parse_reasoning_response(raw: str) -> ParsedResponse:
    if not raw:
        return ParsedResponse(raw="", thinking=None, answer="", has_trace=False,
                              truncated_trace=False, token_estimate=0)

    text = raw

    # Try closed `<think>...</think>` first
    for pat in _THINK_BLOCK_PATTERNS:
        match = pat.search(text)
        if match:
            thinking = match.group(1).strip()
            answer = (text[: match.start()] + text[match.end() :]).strip()
            return ParsedResponse(
                raw=raw,
                thinking=thinking,
                answer=answer,
                has_trace=True,
                truncated_trace=False,
                token_estimate=_token_estimate(answer),
            )

    # Open `<think>` without close — typically max-tokens cut us off
    for pat in _THINK_OPEN_TAILS:
        match = pat.search(text)
        if match:
            thinking_partial = match.group(1).strip()
            answer = text[: match.start()].strip()
            return ParsedResponse(
                raw=raw,
                thinking=thinking_partial,
                answer=answer,
                has_trace=True,
                truncated_trace=True,
                token_estimate=_token_estimate(answer),
            )

    return ParsedResponse(
        raw=raw,
        thinking=None,
        answer=raw.strip(),
        has_trace=False,
        truncated_trace=False,
        token_estimate=_token_estimate(raw),
    )


def reasoning_aware_prompt_prefix(enable: bool) -> str:
    """Returns a system-prompt fragment that asks the model to produce a
    reasoning trace before its answer. Use only when (a) the base model
    supports it without significant degradation, or (b) downstream code
    is ready to consume thinking blocks.

    The prefix is intentionally short so it adds minimal token overhead
    when enabled. Models that don't support `<think>` will mostly ignore
    the instruction, which is fine — the parser handles absence.
    """
    if not enable:
        return ""
    return (
        "Before answering, place your reasoning inside `<think>...</think>` tags. "
        "Be specific about evidence you considered and assumptions you made. "
        "After the closing `</think>`, give your final answer. "
        "If the question is trivial, you may omit the think block.\n\n"
    )


def _token_estimate(text: str) -> int:
    """Rough token count (≈ words × 1.33)."""
    if not text:
        return 0
    return int(len(text.split()) * 1.33)
